//! Google Congestion Control (GoogCC) Bandwidth Estimation based on TWCC feedback.
//!
//! This implementation is ported from libWebRTC's GoogCC and goes beyond the simplified
//! IETF draft (<https://datatracker.ietf.org/doc/html/draft-ietf-rmcat-gcc-02>) to include
//! WebRTC's production features:
//!
//! - Delay-based control (trendline estimator with AIMD rate control)
//! - Loss-based control (with inherent loss rate estimation)
//! - Probe controller with state machine and multi-stage probing strategy
//! - ALR (Application Limited Region) detection and periodic probing
//! - Link capacity estimation from ALR probes
//!
//! The probe controller in particular closely matches WebRTC's `ProbeController` behavior
//! and default constants, enabling compatible bandwidth discovery with WebRTC endpoints.

use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

use crate::Reason;
use crate::rtp_::{Bitrate, DataSize, TwccClusterId, TwccSendRecord, TwccSeq};
use crate::util::Soonest;

mod acked_bitrate_estimator;
mod alr_detector;
pub(crate) mod api;
mod delay;
mod link_capacity_estimator;
mod loss_controller;
mod macros;
mod probe;
mod smoother;
mod time;

use acked_bitrate_estimator::AckedBitrateEstimator;
use alr_detector::AlrDetector;
use delay::DelayController;
use link_capacity_estimator::LinkCapacityEstimator;
use loss_controller::{LossController, LossControllerState};
use macros::log_loss;
use smoother::EstimateSmoother;

pub(crate) use macros::{log_pacer_media_debt, log_pacer_padding_debt};
#[cfg(test)]
pub(crate) use probe::ProbeKind;
pub(crate) use probe::{BandwidthLimitedCause, ProbeEstimator};
pub(crate) use probe::{ProbeClusterState, ProbeControl};

#[cfg(feature = "_internal_test_exports")]
pub use probe::ProbeClusterConfig;
#[cfg(not(feature = "_internal_test_exports"))]
pub(crate) use probe::ProbeClusterConfig;

const INITIAL_BITRATE_WINDOW: Duration = Duration::from_millis(500);
const BITRATE_WINDOW: Duration = Duration::from_millis(150);
const STARTUP_PHASE: Duration = Duration::from_secs(2);
const SEND_RATE_WINDOW: Duration = Duration::from_millis(400);
const SEND_RATE_QUEUE_GROWTH: Duration = Duration::from_millis(8);
const SEND_RATE_CHECK: Duration = Duration::from_millis(125);
const SEND_RATE_RETRY: Duration = Duration::from_secs(5);
const SEND_RATE_IDLE: Duration = Duration::from_secs(1);
const SEND_RATE_STEP: f64 = 1.16;
const DELIVERY_TRIAL_HEADROOM: f64 = 1.65;
const PROBED_DELIVERY_HEADROOM: f64 = 2.5;
const DELIVERY_RESUME_HEADROOM: f64 = 1.4;
const PROBE_HEADROOM: f64 = 1.2;

pub struct Bwe {
    bwe: SendSideBandwidthEstimator,
    desired_bitrate: Bitrate,
    minimum_bitrate: Bitrate,
    smoother: EstimateSmoother,
    sustained_delivery: bool,
    send_rate: Bitrate,
    delivery: Option<DeliveryWindow>,
    confirmed: Option<Bitrate>,
    active_delivery: Option<(Instant, Bitrate)>,
    trial_base: Option<Bitrate>,
    next_trial: Option<Instant>,
}

impl Bwe {
    pub fn new(initial: Bitrate) -> Self {
        let send_side_bwe = SendSideBandwidthEstimator::new(initial);
        Bwe {
            bwe: send_side_bwe,
            desired_bitrate: Bitrate::ZERO,
            minimum_bitrate: Bitrate::ZERO,
            smoother: EstimateSmoother::new(),
            sustained_delivery: false,
            send_rate: initial,
            delivery: None,
            confirmed: None,
            active_delivery: None,
            trial_base: None,
            next_trial: None,
        }
    }

    pub fn with_sustained_delivery(mut self, enabled: bool) -> Self {
        self.sustained_delivery = enabled;
        self
    }

    pub fn handle_timeout(&mut self, now: Instant, do_probe: bool) -> Option<ProbeClusterConfig> {
        if self.sustained_delivery
            && self
                .bwe
                .started_at
                .is_some_and(|start| now.saturating_duration_since(start) > STARTUP_PHASE)
        {
            self.bwe
                .probe_control
                .set_ceiling((self.send_rate * PROBE_HEADROOM).max(Bitrate::mbps(1)));
        }
        let result = self.bwe.handle_timeout(self.desired_bitrate, do_probe, now);
        if self.sustained_delivery {
            self.update_send_rate(None, None, now);
        } else if let Some(rate) = self.bwe.last_estimate() {
            self.smoother.record(now, rate);
        }
        result
    }

    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.bwe.start_probe(config, now)
    }

    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.bwe.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        self.bwe.reset(init_bitrate);
        if self.sustained_delivery {
            self.send_rate = init_bitrate;
            self.delivery = None;
            self.confirmed = None;
            self.active_delivery = None;
            self.trial_base = None;
            self.next_trial = None;
            self.smoother = EstimateSmoother::new();
        }
    }

    pub fn update<'t>(
        &mut self,
        records: impl Iterator<Item = &'t crate::rtp_::TwccSendRecord>,
        now: Instant,
    ) {
        if !self.sustained_delivery {
            self.bwe.update(records, now);
            return;
        }

        let mut delivered = None;
        let mut ordinary_packets = 0;
        let mut lost_packets = 0;
        let probe = self.bwe.update(
            records.inspect(|record| {
                if record.cluster().is_some() {
                    return;
                }
                ordinary_packets += 1;
                if let Some(at) = record.remote_recv_time() {
                    delivered = record_delivery(
                        &mut self.delivery,
                        record.local_send_time(),
                        at,
                        record.size().into(),
                        self.send_rate,
                    )
                    .or(delivered);
                } else {
                    lost_packets += 1;
                }
            }),
            now,
        );
        if lost_packets >= 2 && lost_packets * 50 >= ordinary_packets {
            delivered = Some(Delivery::Congested);
            self.delivery = None;
        }
        self.update_send_rate(delivered, probe, now);
    }

    fn update_send_rate(&mut self, sample: Option<Delivery>, probe: Option<Bitrate>, now: Instant) {
        let Some(estimate) = self.bwe.last_estimate() else {
            return;
        };
        if in_startup_phase(self.bwe.started_at, now) || self.desired_bitrate.is_zero() {
            self.send_rate = estimate;
            self.delivery = None;
            self.trial_base = None;
        } else {
            let mut delivery_cap = Bitrate::ZERO;
            match sample {
                Some(Delivery::Congested) => {
                    // Leave room for the queue to drain and delay retrying a failed trial.
                    let trial = self.trial_base.take();
                    self.send_rate =
                        (trial.unwrap_or(self.send_rate) * 0.85).max(Bitrate::kbps(40));
                    self.confirmed = None;
                    self.active_delivery = None;
                    if trial.is_some() {
                        self.next_trial = Some(now + SEND_RATE_RETRY);
                    }
                }
                Some(Delivery::Rate(delivered))
                    if delivered * PROBED_DELIVERY_HEADROOM >= self.send_rate =>
                {
                    self.active_delivery = Some((now, delivered));
                    delivery_cap = delivered * DELIVERY_TRIAL_HEADROOM;
                    let supported = self.send_rate.min(delivered * DELIVERY_RESUME_HEADROOM);
                    self.confirmed =
                        Some(self.confirmed.map_or(supported, |rate| rate.max(supported)));
                    if delivery_cap >= self.send_rate {
                        self.trial_base = None;
                    }
                }
                _ => {}
            }
            let active = self
                .active_delivery
                .filter(|(at, _)| now.saturating_duration_since(*at) < SEND_RATE_IDLE);
            if let Some((_, delivered)) = active {
                // An encoder can underfill its budget. A fresh probe lets it try
                // a small increase during motion, with a cap based on media use.
                let probe_cap = probe.map_or(Bitrate::ZERO, |rate| {
                    (rate * 0.95).min(delivered * PROBED_DELIVERY_HEADROOM)
                });
                if self.next_trial.is_none_or(|retry| now >= retry) {
                    let target = (self.send_rate * SEND_RATE_STEP)
                        .min(delivery_cap.max(probe_cap))
                        .min(self.desired_bitrate);
                    if target > self.send_rate {
                        self.trial_base = Some(self.send_rate);
                        self.send_rate = target;
                    }
                }
            } else if let Some(confirmed) = self.confirmed
                && confirmed < self.send_rate
            {
                // Resume at the last supported rate after a quiet period.
                self.send_rate = confirmed;
                self.trial_base = None;
            }
        }
        self.smoother
            .record_for(now, self.send_rate, SEND_RATE_IDLE);
    }

    pub fn poll_estimate(&mut self) -> Option<Bitrate> {
        self.smoother.poll()
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        self.bwe.poll_timeout()
    }

    pub fn last_estimate(&self) -> Option<Bitrate> {
        if self.sustained_delivery {
            Some(self.send_rate)
        } else {
            self.bwe.last_estimate()
        }
    }

    pub fn on_media_sent(&mut self, payload_size: DataSize, is_padding: bool, now: Instant) {
        if !is_padding {
            // Update ALR detector with media bytes sent
            self.bwe.on_media_sent(payload_size, now);
        }
    }

    pub fn is_overusing(&self) -> bool {
        self.bwe.is_overusing()
    }

    pub fn set_bitrate(&mut self, minimum: Bitrate, desired: Bitrate) {
        self.minimum_bitrate = minimum.min(desired);
        self.desired_bitrate = desired;
    }

    pub fn minimum_bitrate(&self) -> Bitrate {
        self.minimum_bitrate
    }
}

struct DeliveryWindow {
    sent: Instant,
    received: Instant,
    bytes: DataSize,
    queue_grew: bool,
}

#[derive(Clone, Copy)]
enum Delivery {
    Rate(Bitrate),
    Congested,
}

fn record_delivery(
    window: &mut Option<DeliveryWindow>,
    sent: Instant,
    received: Instant,
    bytes: DataSize,
    send_rate: Bitrate,
) -> Option<Delivery> {
    let previous = window.get_or_insert(DeliveryWindow {
        sent,
        received,
        bytes: DataSize::ZERO,
        queue_grew: false,
    });
    // Feedback can arrive out of order.
    let send_span = sent.checked_duration_since(previous.sent)?;
    let receive_span = received.checked_duration_since(previous.received)?;
    // A comparison of relative transit times works even when the two clocks
    // have different origins. Check rising delay before a full rate sample.
    let queue_grew = send_span >= SEND_RATE_CHECK
        && receive_span > send_span + SEND_RATE_QUEUE_GROWTH
        && previous.bytes / send_span >= send_rate * 0.35;
    let sample = if queue_grew {
        // Confirm increasing delay in adjacent windows before reducing the rate.
        previous.queue_grew.then_some(Delivery::Congested)
    } else if send_span >= SEND_RATE_WINDOW {
        Some(Delivery::Rate(previous.bytes / send_span.max(receive_span)))
    } else {
        previous.bytes += bytes;
        return None;
    };
    *previous = DeliveryWindow {
        sent,
        received,
        bytes,
        queue_grew,
    };
    sample
}

struct SendSideBandwidthEstimator {
    delay_controller: DelayController,
    loss_controller: LossController,
    acked_bitrate_estimator: AckedBitrateEstimator,
    probe_control: ProbeControl,
    probe_estimator: ProbeEstimator,
    // First nonempty transport feedback; starts loss estimation and enables ALR probing.
    started_at: Option<Instant>,
    alr_detector: AlrDetector,
    link_capacity_estimator: LinkCapacityEstimator,
    last_updated_estimate: Option<Bitrate>,
}

impl SendSideBandwidthEstimator {
    pub fn new(initial_bitrate: Bitrate) -> Self {
        let mut alr_detector = AlrDetector::new();
        alr_detector.set_estimated_bitrate(initial_bitrate);

        let mut loss_controller = LossController::new();
        loss_controller.set_bandwidth_estimate(initial_bitrate);

        Self {
            delay_controller: DelayController::new(initial_bitrate),
            loss_controller,
            acked_bitrate_estimator: AckedBitrateEstimator::new(
                INITIAL_BITRATE_WINDOW,
                BITRATE_WINDOW,
            ),
            probe_control: ProbeControl::new(),
            probe_estimator: ProbeEstimator::new(),
            started_at: None,
            alr_detector,
            link_capacity_estimator: LinkCapacityEstimator::new(),
            last_updated_estimate: None,
        }
    }

    /// Whether the delay-based detector currently signals overuse.
    ///
    /// This is useful for gating behaviors (like padding/probing) that would otherwise
    /// re-excite the system while we're already congested.
    pub fn is_overusing(&self) -> bool {
        self.delay_controller.is_overusing()
    }

    /// Update ALR detector with actual bytes sent.
    ///
    /// Should be called for media packets (not padding/probes).
    /// This is typically called from the session's packet sending logic.
    pub fn on_media_sent(&mut self, bytes: DataSize, now: Instant) {
        self.alr_detector.on_bytes_sent(bytes, now);
    }

    /// Record a packet from a TWCC report.
    pub fn update<'t>(
        &mut self,
        records: impl Iterator<Item = &'t TwccSendRecord>,
        now: Instant,
    ) -> Option<Bitrate> {
        let send_records: Vec<_> = records.collect();
        if send_records.is_empty() {
            return None;
        }
        let _ = self.started_at.get_or_insert(now);

        // Feed records to probe estimator for analysis and process any new probe results
        let mut latest_probe_result = None;
        let mut clean_probe = None;
        let previous_estimate = self.last_estimate();
        for (config, bitrate, saturated) in
            self.probe_estimator.update(send_records.iter().copied())
        {
            if !saturated {
                clean_probe = Some(bitrate.min(config.target_bitrate()));
            }
            // A probe the receiver kept up with only establishes a lower bound.
            if !saturated && previous_estimate.is_some_and(|previous| bitrate < previous) {
                continue;
            }

            latest_probe_result = Some(bitrate);

            // Update link capacity estimator for every successful ALR probe, not just the latest.
            // The estimator internally takes the max of all probe results, building up knowledge
            // of proven link capacity. This differs from the delay controller, which only receives
            // the latest probe result (matching WebRTC's FetchAndResetLastEstimatedBitrate behavior).
            if config.is_alr_probe() {
                self.link_capacity_estimator.update_from_probe(bitrate, now);
            }
        }

        let mut acked_packets = vec![];

        let mut max_rtt = None;
        let mut count = 0;
        let mut lost = 0;
        for record in send_records.iter() {
            count += 1;
            let Ok(acked_packet) = (*record).try_into() else {
                lost += 1;
                continue;
            };
            acked_packets.push(acked_packet);
            max_rtt = max_rtt.max(record.rtt());
        }
        acked_packets.sort_by(AckedPacket::order_by_receive_time);

        for acked_packet in acked_packets.iter() {
            self.acked_bitrate_estimator
                .update(acked_packet.remote_recv_time, acked_packet.size);
        }

        let acked_bitrate = self.acked_bitrate_estimator.current_estimate();

        // Use the latest probe result from this update, if any
        let probe_result = latest_probe_result;

        let is_probe_result = probe_result.is_some();

        // Update delay controller with the latest probe result
        let maybe_estimate =
            self.delay_controller
                .update(&acked_packets, acked_bitrate, probe_result, now);

        let Some(delay_estimate) = maybe_estimate else {
            return clean_probe;
        };

        let loss = if count == 0 {
            0.0
        } else {
            lost as f64 / count as f64
        };
        log_loss!(loss);

        // During startup with no loss, use delay-based estimate directly
        if in_startup_phase(self.started_at, now) && loss <= 0.001 {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
            return clean_probe;
        }

        // When probe succeeds, set bandwidth directly
        if is_probe_result {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
        }

        if let Some(acked_bitrate) = acked_bitrate {
            self.loss_controller.set_acknowledged_bitrate(acked_bitrate);
        }

        // This corresponds to UpdateLossBasedEstimator + UpdateEstimate
        self.loss_controller
            .update_bandwidth_estimate(&send_records, delay_estimate);

        // Loss-based result is capped by delay_based_limit
        let loss_result = self.loss_controller.loss_based_result();
        if let Some(loss_estimate) = loss_result.bandwidth_estimate {
            if loss_estimate > delay_estimate {
                // Loss controller produced higher estimate than delay controller
                // Cap it at delay estimate (delay controller is the upper limit)
                self.loss_controller.set_bandwidth_estimate(delay_estimate);
            }
        }

        // Feed the (possibly combined) estimate into subcomponents wanting it.
        self.propagate_estimate();

        clean_probe
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        let delay_timeout = Some(self.delay_controller.poll_timeout());
        let probe_timeout = Some(self.probe_control.poll_timeout());
        let probe_estimator_timeout = Some(self.probe_estimator.poll_timeout());
        (delay_timeout, Reason::BweDelayControl)
            .soonest((probe_timeout, Reason::BweProbeControl))
            .soonest((probe_estimator_timeout, Reason::BweProbeEstimator))
    }

    /// Handle periodic timeout for BWE components.
    pub fn handle_timeout(
        &mut self,
        desired_bitrate: Bitrate,
        do_probe: bool,
        now: Instant,
    ) -> Option<ProbeClusterConfig> {
        self.delay_controller
            .handle_timeout(self.acked_bitrate_estimator.current_estimate(), now);

        // Update probe control with desired bitrate.
        self.probe_control.set_desired_bitrate(desired_bitrate);

        // Empty media queues are application-limited too. Advance the budget
        // without charging probe/padding bytes, including before the first media.
        if do_probe {
            self.alr_detector.handle_timeout(now);
        }

        // Get ALR state and forward to both probe control and loss controller
        let alr_start_time = self.alr_detector.alr_start_time();
        // ALR probing requires feedback for at least one sent packet.
        if let Some(t) = alr_start_time.filter(|_| self.started_at.is_some()) {
            self.probe_control.set_alr_start_time(t);
        } else {
            self.probe_control.set_alr_stop_time(now);
        }

        self.loss_controller.set_alr_start_time(alr_start_time);

        // Get link capacity estimate and forward to loss controller.
        let link_capacity = self.link_capacity_estimator.capacity_estimate(now);
        self.loss_controller
            .set_link_capacity_estimate(link_capacity);

        // Clean up expired probe cluster state
        self.probe_estimator.handle_timeout(now);

        // Feed the current estimate into subcontrollers, if it changed.
        self.propagate_estimate();

        // If we can't probe, clear any pending/active probes
        if !do_probe {
            self.probe_estimator.clear_probes();
        }

        self.probe_control.enable(do_probe);

        // Timeout-driven probe logic (WebRTC `Process()` equivalent).
        self.probe_control.handle_timeout(now)
    }

    fn propagate_estimate(&mut self) {
        // Do we have a value?
        let Some(estimate) = self.last_estimate() else {
            return;
        };
        // Did it change?
        if self.last_updated_estimate == Some(estimate) {
            return;
        }

        let cause = self.bandwidth_limited_cause();

        self.probe_control.set_estimated_bitrate(estimate, cause);
        self.alr_detector.set_estimated_bitrate(estimate);

        // Don't update until this changes.
        self.last_updated_estimate = Some(estimate);
    }

    fn bandwidth_limited_cause(&self) -> BandwidthLimitedCause {
        if self.delay_controller.is_overusing() {
            return BandwidthLimitedCause::DelayBasedLimitedDelayIncreased;
        }

        match self.loss_controller.loss_based_result().state {
            LossControllerState::DelayBased => BandwidthLimitedCause::DelayBasedLimited,
            LossControllerState::Increasing => BandwidthLimitedCause::LossLimitedBweIncreasing,
            LossControllerState::Decreasing => BandwidthLimitedCause::LossLimitedBwe,
        }
    }

    /// Get the latest estimate.
    pub fn last_estimate(&self) -> Option<Bitrate> {
        let delay_estimate = self.delay_controller.last_estimate();

        let loss_result = self.loss_controller.loss_based_result();

        // Only apply loss-based limiting when actively in a loss-limiting state
        match loss_result.state {
            LossControllerState::DelayBased => {
                // Loss controller defers to delay-based estimate
                delay_estimate
            }
            LossControllerState::Decreasing | LossControllerState::Increasing => {
                // Loss controller is actively limiting or recovering
                match (delay_estimate, loss_result.bandwidth_estimate) {
                    (Some(de), Some(le)) => Some(de.min(le)),
                    (None, le @ Some(_)) => le,
                    (de @ Some(_), None) => de,
                    (None, None) => None,
                }
            }
        }
    }

    /// Start analyzing a probe cluster.
    ///
    /// This should be called when the pacer starts sending a probe cluster,
    /// to tell the estimator which cluster to watch for in TWCC feedback.
    /// Returns `true` if the probe was started, `false` if rejected.
    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.probe_estimator.probe_start(config, now)
    }

    /// End a probe cluster and mark it for cleanup.
    ///
    /// This should be called when the pacer finishes sending a probe cluster.
    /// The estimator will continue collecting feedback for a cluster history period
    /// to allow late-arriving TWCC reports to refine the estimate.
    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.probe_estimator.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        *self = Self::new(init_bitrate);
    }
}

/// A RTP packet that has been sent and acknowledged by the receiver in a TWCC report.
#[derive(Debug, Copy, Clone)]
pub struct AckedPacket {
    /// The TWCC sequence number
    seq_no: TwccSeq,
    /// The size of the packets in bytes.
    size: DataSize,
    /// When we sent the packet
    local_send_time: Instant,
    /// When the packet was received at the remote, note this Instant is only usable with other
    /// instants of the same type i.e. those that represent a TWCC reported receive time for this
    /// session.
    remote_recv_time: Instant,
    /// The local time when received confirmation that the other side received the seq i.e. when we
    /// received the TWCC report for this packet.
    local_recv_time: Instant,
}

impl AckedPacket {
    fn rtt(&self) -> Duration {
        self.local_recv_time - self.local_send_time
    }

    fn order_by_receive_time(lhs: &Self, rhs: &Self) -> Ordering {
        if lhs.remote_recv_time != rhs.remote_recv_time {
            lhs.remote_recv_time.cmp(&rhs.remote_recv_time)
        } else if lhs.local_send_time != rhs.local_send_time {
            lhs.local_send_time.cmp(&rhs.local_send_time)
        } else {
            lhs.seq_no.cmp(&rhs.seq_no)
        }
    }
}

// NB: Extracted for lifetime reasons
fn in_startup_phase(started_at: Option<Instant>, now: Instant) -> bool {
    started_at
        .map(|s| now.duration_since(s) <= STARTUP_PHASE)
        .unwrap_or(false)
}

impl TryFrom<&TwccSendRecord> for AckedPacket {
    type Error = ();

    fn try_from(value: &TwccSendRecord) -> Result<Self, Self::Error> {
        let Some(remote_recv_time) = value.remote_recv_time() else {
            return Err(());
        };
        let Some(local_recv_time) = value.local_recv_time() else {
            return Err(());
        };

        Ok(Self {
            seq_no: value.seq(),
            size: value.size().into(),
            local_send_time: value.local_send_time(),
            remote_recv_time,
            local_recv_time,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandwidthUsage {
    Overuse,
    Normal,
    Underuse,
}

impl fmt::Display for BandwidthUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthUsage::Overuse => write!(f, "overuse"),
            BandwidthUsage::Normal => write!(f, "normal"),
            BandwidthUsage::Underuse => write!(f, "underuse"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rtp_::TwccPacketId;

    #[test]
    fn unsaturated_probes_establish_a_lower_bound() {
        use crate::bwe_::probe::ProbeKind;

        for (previous_mbps, receive_interval, expected_kbps) in
            [(10, 4, 10_000), (1, 4, 2_400), (10, 8, 1_140)]
        {
            let start = Instant::now();
            let mut bwe = SendSideBandwidthEstimator::new(Bitrate::mbps(previous_mbps));
            let probe =
                ProbeClusterConfig::new(1.into(), Bitrate::mbps(15), ProbeKind::PeriodicAlr);
            assert!(bwe.start_probe(probe, start));
            let feedback_at = start + Duration::from_millis(300);
            let records: Vec<_> = (0..32_u64)
                .map(|i| {
                    TwccSendRecord::test_new(
                        TwccPacketId::with_cluster(i, probe.cluster()),
                        start + Duration::from_millis(i * 4),
                        1200,
                        feedback_at,
                        Some(start + Duration::from_millis(10 + i * receive_interval)),
                    )
                })
                .collect();
            bwe.update(records.iter(), feedback_at);
            let estimate = bwe.last_estimate().unwrap();
            assert_eq!(estimate.as_u64(), expected_kbps * 1000);
        }
    }

    #[test]
    fn alr_probing_waits_for_nonempty_transport_feedback() {
        let start = Instant::now();
        let desired = Bitrate::mbps(3);
        let mut bwe = SendSideBandwidthEstimator::new(Bitrate::kbps(300));

        // Initial discovery must work before feedback or media exists.
        assert!(bwe.handle_timeout(desired, true, start).is_some());
        assert!(bwe.handle_timeout(desired, true, start).is_some());
        for second in 1..=7 {
            let now = start + Duration::from_secs(second);
            bwe.update(std::iter::empty(), now);
            assert!(
                bwe.handle_timeout(desired, true, now).is_none(),
                "empty feedback must not authorize periodic ALR probes"
            );
        }
        assert!(bwe.alr_detector.alr_start_time().is_some());
        assert!(bwe.started_at.is_none());

        // Feedback for an actual sent packet authorizes ALR discovery.
        let now = start + Duration::from_secs(8);
        let record = TwccSendRecord::test_new(
            TwccPacketId::new(1u64),
            now - Duration::from_millis(20),
            240,
            now,
            Some(now - Duration::from_millis(10)),
        );
        bwe.update(std::iter::once(&record), now);
        let probe = bwe.handle_timeout(desired, true, now).unwrap();
        assert!(probe.is_alr_probe());

        // A reset starts discovery afresh, including the feedback requirement.
        bwe.reset(Bitrate::kbps(300));
        assert!(bwe.handle_timeout(desired, true, now).is_some());
        assert!(bwe.handle_timeout(desired, true, now).is_some());
        for second in 1..=7 {
            assert!(
                bwe.handle_timeout(desired, true, now + Duration::from_secs(second))
                    .is_none()
            );
        }
    }

    #[test]
    fn bitrate_targets_survive_estimator_reset() {
        let mut bwe = Bwe::new(Bitrate::kbps(300));
        bwe.set_bitrate(Bitrate::kbps(800), Bitrate::mbps(3));
        bwe.reset(Bitrate::kbps(200));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::kbps(800));
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(3));
        assert_eq!(bwe.last_estimate(), Some(Bitrate::kbps(200)));
    }

    #[test]
    fn minimum_is_clamped_to_desired_and_can_be_cleared() {
        let mut bwe = Bwe::new(Bitrate::kbps(300));
        bwe.set_bitrate(Bitrate::mbps(2), Bitrate::mbps(1));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::mbps(1));
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(1));
        bwe.set_bitrate(Bitrate::ZERO, Bitrate::mbps(1));
        assert_eq!(bwe.minimum_bitrate(), Bitrate::ZERO);
        assert_eq!(bwe.desired_bitrate, Bitrate::mbps(1));
    }
}
