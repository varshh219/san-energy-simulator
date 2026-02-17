
# Run with:
#    streamlit run app.py
#
# Requirements:
#    streamlit
#    numpy
#    pandas
#    matplotlib

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import textwrap
from dataclasses import dataclass, field

st.set_page_config(page_title="SAN Energy Simulator (GUI)", layout="wide")


class Disk:
    STATE_ACTIVE = "ACTIVE"
    STATE_IDLE = "IDLE"
    STATE_STANDBY = "STANDBY"

    def __init__(self,
                 active_power=6.0,
                 idle_power=3.0,
                 standby_power=0.3,
                 spin_up_time=5.0,
                 spin_down_time=2.0,
                 spin_up_energy=10.0,
                 spin_down_energy=2.0):
        self.active_power = float(active_power)
        self.idle_power = float(idle_power)
        self.standby_power = float(standby_power)
        self.spin_up_time = float(spin_up_time)
        self.spin_down_time = float(spin_down_time)
        self.spin_up_energy = float(spin_up_energy)
        self.spin_down_energy = float(spin_down_energy)
        self.reset()

    def reset(self):
        self.state = Disk.STATE_ACTIVE
        self.time_in_state = {Disk.STATE_ACTIVE: 0.0,
                              Disk.STATE_IDLE: 0.0,
                              Disk.STATE_STANDBY: 0.0}
        self.energy_spent = 0.0
        self.last_time = 0.0
        self.last_request = None
        self.spin_count_up = 0
        self.spin_count_down = 0

    def _accumulate(self, from_t, to_t):
        if to_t <= from_t:
            return
        duration = to_t - from_t
        self.time_in_state[self.state] += duration
        if self.state == Disk.STATE_ACTIVE:
            self.energy_spent += self.active_power * duration
        elif self.state == Disk.STATE_IDLE:
            self.energy_spent += self.idle_power * duration
        elif self.state == Disk.STATE_STANDBY:
            self.energy_spent += self.standby_power * duration

    def advance_to(self, t):
        self._accumulate(self.last_time, t)
        self.last_time = t

    def spin_down(self, t):
        self.advance_to(t)
        self.energy_spent += self.spin_down_energy
        self.state = Disk.STATE_STANDBY
        self.last_time = t
        self.spin_count_down += 1

    def spin_up_and_serve(self, t):
        latency = 0.0
        if self.state == Disk.STATE_STANDBY:
            self.advance_to(t)
            self.energy_spent += self.spin_up_energy
            self.time_in_state[Disk.STATE_ACTIVE] += self.spin_up_time
            self.energy_spent += self.active_power * self.spin_up_time
            latency += self.spin_up_time
            self.spin_count_up += 1
            self.state = Disk.STATE_ACTIVE
            self.last_time = t + latency
            self.last_request = t + latency
            return latency
        self.advance_to(t)
        self.state = Disk.STATE_ACTIVE
        self.last_time = t
        self.last_request = t
        return latency

    def maybe_idle(self):
        if self.state == Disk.STATE_ACTIVE:
            self.state = Disk.STATE_IDLE

    def finalize(self, t_end):
        self.advance_to(t_end)

    def get_summary(self):
        return {
            "time_in_state": dict(self.time_in_state),
            "energy_spent": float(self.energy_spent),
            "spin_up_count": int(self.spin_count_up),
            "spin_down_count": int(self.spin_count_down),
            "final_state": self.state
        }

# -------------------------
# Workload generator
# -------------------------
def generate_workload(kind="random", num_ops=500, duration=600, read_ratio=0.7, seed=None,
                      gap_min=0.0, gap_max=15.0):
    rng = np.random.default_rng(seed)
    if kind == "random":
        gaps = rng.uniform(gap_min, gap_max, size=num_ops)
        times = np.cumsum(gaps)
        times = times[times <= duration]
    elif kind == "sequential":
        times = np.linspace(0, duration, num_ops)
    elif kind == "bursty":
        centers = rng.uniform(0, duration, size=max(1, num_ops // 50))
        times = []
        for i in range(num_ops):
            c = centers[rng.integers(0, len(centers))]
            times.append(min(max(rng.normal(c, duration * 0.01), 0), duration))
        times = np.sort(times)
    else:
        times = np.sort(rng.uniform(0, duration, size=num_ops))

    ops = []
    for t in times:
        ops.append((float(t), 'R' if rng.random() < read_ratio else 'W'))
    return ops

# -------------------------
# Predictors
# -------------------------
@dataclass
class MovingAveragePredictor:
    window: int = 5
    arrivals: list = field(default_factory=list)

    def update(self, arrival_time):
        self.arrivals.append(arrival_time)
        if len(self.arrivals) > self.window + 1:
            self.arrivals = self.arrivals[-(self.window + 1):]

    def predict_next_interarrival(self):
        if len(self.arrivals) < 2:
            return None
        inters = [self.arrivals[i] - self.arrivals[i-1] for i in range(1, len(self.arrivals))]
        k = min(len(inters), self.window)
        return float(np.mean(inters[-k:]))

    def reset(self):
        self.arrivals = []

@dataclass
class EWMAPredictor:
    alpha: float = 0.3
    last_ewma: float = None
    last_arrival: float = None

    def update(self, arrival_time):
        if self.last_arrival is None:
            self.last_arrival = arrival_time
            return
        ia = arrival_time - self.last_arrival
        if self.last_ewma is None:
            self.last_ewma = ia
        else:
            self.last_ewma = self.alpha * ia + (1 - self.alpha) * self.last_ewma
        self.last_arrival = arrival_time

    def predict_next_interarrival(self):
        return None if self.last_ewma is None else float(self.last_ewma)

    def reset(self):
        self.last_ewma = None
        self.last_arrival = None

# -------------------------
# Simulation policies
# -------------------------
def run_policy(workload, disk_params, predictor_type, predictor_params, idle_threshold, policy_type, aggressive_factor=0.5):
    disk = Disk(**disk_params)
    if predictor_type == "moving":
        pred = MovingAveragePredictor(**predictor_params)
    else:
        pred = EWMAPredictor(**predictor_params)

    requests_log = []
    disk.reset()
    pred.reset()

    for idx, (t, op) in enumerate(workload):
        ref = disk.last_request if disk.last_request is not None else disk.last_time
        idle_time = t - ref

        # Baseline
        if policy_type == "baseline":
            if idle_time >= idle_threshold and disk.state != Disk.STATE_STANDBY:
                disk.spin_down(t - 1e-9)

        # Predictive (guaranteed spin-down)
        elif policy_type == "predictive":
            predicted = pred.predict_next_interarrival()
            if predicted is not None:
                spin_time = (disk.last_request if disk.last_request is not None else disk.last_time) + idle_threshold * aggressive_factor
                if disk.state != Disk.STATE_STANDBY and spin_time < t:
                    disk.spin_down(spin_time)

        else:
            if idle_time >= idle_threshold and disk.state != Disk.STATE_STANDBY:
                disk.spin_down(t - 1e-9)

        # Serve request
        latency = disk.spin_up_and_serve(t)
        requests_log.append({"arrival": t, "op": op, "latency": latency, "state_after": disk.state})
        pred.update(t)
        disk.maybe_idle()

    end_time = (workload[-1][0] + 1.0) if workload else 1.0
    disk.finalize(end_time)
    return {"disk_summary": disk.get_summary(), "requests": requests_log, "end_time": end_time}

# -------------------------
# Compare policies
# -------------------------
def compare_policies(workload, disk_params, predictor_choice, predictor_params, idle_threshold, aggressive_factor):
    baseline = run_policy(workload, disk_params, predictor_choice, predictor_params, idle_threshold, "baseline", aggressive_factor=1.0)
    predictive = run_policy(workload, disk_params, predictor_choice, predictor_params, idle_threshold, "predictive", aggressive_factor=aggressive_factor)
    return {"baseline": baseline, "predictive": predictive}

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔋 SAN Energy Simulator — (Predictive vs Baseline)")
st.markdown("**Use Gap controls and Idle Threshold to force visible idle periods.** ")

# Sidebar
with st.sidebar:
    st.header("Workload Settings")
    wtype = st.selectbox("Workload Type", ["random", "sequential", "bursty"])
    num_ops = st.slider("Number of operations", 10, 5000, 300)
    duration = st.slider("Simulation duration (s)", 30, 3600, 600)
    read_ratio = st.slider("Read ratio", 0.0, 1.0, 0.7)
    st.subheader("Idle-gap controls")
    gap_min = st.number_input("Gap min (s)", min_value=0.0, max_value=300.0, value=2.0)
    gap_max = st.number_input("Gap max (s)", min_value=0.0, max_value=300.0, value=30.0)

    st.header("Disk Settings")
    active_power = st.number_input("Active Power (W)", value=6.0, min_value=0.0)
    idle_power = st.number_input("Idle Power (W)", value=3.0, min_value=0.0)
    standby_power = st.number_input("Standby Power (W)", value=0.3, min_value=0.0)
    spin_up_time = st.number_input("Spin-up time (s)", value=5.0, min_value=0.0)
    spin_down_time = st.number_input("Spin-down time (s)", value=2.0, min_value=0.0)
    spin_up_energy = st.number_input("Spin-up energy (J)", value=10.0, min_value=0.0)
    spin_down_energy = st.number_input("Spin-down energy (J)", value=2.0, min_value=0.0)

    st.header("Predictor Settings")
    predictor_choice = st.selectbox("Predictor type", ["moving", "ewma"])
    if predictor_choice == "moving":
        ma_window = st.slider("Moving average window (k)", 1, 100, 5)
        predictor_params = {"window": ma_window}
    else:
        ewma_alpha = st.slider("EWMA alpha", 0.01, 0.99, 0.3)
        predictor_params = {"alpha": ewma_alpha}

    st.header("Policy Settings")
    idle_threshold = st.number_input("Idle threshold (s)", value=10.0, min_value=0.1)
    aggressive_factor = st.slider("Aggressive factor", 0.1, 1.0, 0.3)
    run_button = st.button("Run Simulation")

# Main
st.markdown("## Simulation Output")
col_left, col_right = st.columns([2.2, 1])

if run_button:
    workload = generate_workload(kind=wtype, num_ops=num_ops, duration=duration,
                                 read_ratio=read_ratio, seed=None, gap_min=gap_min, gap_max=gap_max)
    if len(workload) == 0:
        st.error("No requests generated. Increase duration or number of operations.")
        st.stop()

    disk_params = {"active_power": active_power, "idle_power": idle_power, "standby_power": standby_power,
                   "spin_up_time": spin_up_time, "spin_down_time": spin_down_time,
                   "spin_up_energy": spin_up_energy, "spin_down_energy": spin_down_energy}

    with st.spinner("Running baseline & predictive policies..."):
        results = compare_policies(workload, disk_params, predictor_choice, predictor_params, idle_threshold, aggressive_factor)

    base_summary = results["baseline"]["disk_summary"]
    pred_summary = results["predictive"]["disk_summary"]
    base_energy = base_summary["energy_spent"]
    pred_energy = pred_summary["energy_spent"]
    saved_pct = (base_energy - pred_energy) / base_energy * 100 if base_energy > 0 else 0.0

    # Left column: timeline, plots
    with col_left:
        st.subheader("Workload (request times)")
        reqs_df = pd.DataFrame(workload, columns=["time", "op"])
        fig_w, axw = plt.subplots(figsize=(9, 1.5))
        axw.scatter(reqs_df["time"], [1]*len(reqs_df), marker='|', s=200)
        axw.set_yticks([])
        axw.set_xlabel("Time (s)")
        axw.set_title("Requests timeline")
        st.pyplot(fig_w)

        st.subheader("Disk State Timeline (Predictive)")
        pred_end = results["predictive"]["end_time"]
        sample_t = np.linspace(0, pred_end, int(min(2000, max(50, pred_end*4))))
        def reconstruct_states(res, sample_t):
            reqs = res["requests"]
            last_req_time = 0.0
            states = []
            req_times = [r["arrival"] for r in reqs]
            for t in sample_t:
                prev = [rt for rt in req_times if rt <= t]
                if len(prev) == 0:
                    states.append(2)
                else:
                    last = prev[-1]
                    if t - last < spin_up_time + 1e-3:
                        states.append(0)
                    elif t - last < idle_threshold:
                        states.append(1)
                    else:
                        states.append(2)
            return states
        states_pred = reconstruct_states(results["predictive"], sample_t)
        fig_s, axs = plt.subplots(figsize=(9, 2))
        axs.plot(sample_t, states_pred, drawstyle='steps-post')
        axs.set_yticks([0,1,2])
        axs.set_yticklabels(["ACTIVE","IDLE","STANDBY"])
        axs.set_xlabel("Time (s)")
        axs.set_title("Approximate Disk State Timeline (Predictive)")
        st.pyplot(fig_s)

        st.subheader("Energy Comparison")
        fig_e, axe = plt.subplots(figsize=(6,3))
        axe.bar(["Baseline", "Predictive"], [base_energy, pred_energy], color=['#d95f02', '#1b9e77'])
        axe.set_ylabel("Energy (J)")
        axe.set_title(f"Total Energy (Predictive saved {saved_pct:.2f}%)")
        st.pyplot(fig_e)

    # Right column: numeric summary and downloads
    with col_right:
        st.subheader("Numeric Summary")
        st.metric("Baseline Energy (J)", f"{base_energy:.2f}")
        st.metric("Predictive Energy (J)", f"{pred_energy:.2f}", delta=f"{-saved_pct:.2f}%")
        st.write("**Baseline disk summary**")
        st.json(base_summary)
        st.write("**Predictive disk summary**")
        st.json(pred_summary)

        st.subheader("Request samples (Predictive)")
        pred_reqs_df = pd.DataFrame(results["predictive"]["requests"])
        if not pred_reqs_df.empty:
            st.dataframe(pred_reqs_df.head(200))
        else:
            st.write("No requests found for predictive policy.")

        # Exports
        st.markdown("---")
        all_rows = []
        for policy_name, res in results.items():
            for r in res["requests"]:
                all_rows.append({
                    "policy": policy_name,
                    "arrival": r["arrival"],
                    "op": r["op"],
                    "latency": r["latency"],
                    "state_after": r["state_after"]
                })
        df_all = pd.DataFrame(all_rows)
        csv_buf = io.StringIO()
        df_all.to_csv(csv_buf, index=False)
        st.download_button("Download requests CSV (all policies)", data=csv_buf.getvalue().encode(), file_name="san_requests_all.csv", mime="text/csv")

        summary_df = pd.DataFrame([
            {"policy": "baseline", "energy_J": base_energy, **base_summary["time_in_state"], "spin_up": base_summary["spin_up_count"], "spin_down": base_summary["spin_down_count"]},
            {"policy": "predictive", "energy_J": pred_energy, **pred_summary["time_in_state"], "spin_up": pred_summary["spin_up_count"], "spin_down": pred_summary["spin_down_count"]}
        ])
        s_buf = io.StringIO()
        summary_df.to_csv(s_buf, index=False)
        st.download_button("Download summary CSV", data=s_buf.getvalue().encode(), file_name="san_summary.csv", mime="text/csv")

        report_text = textwrap.dedent(f"""
        SAN Energy Simulator Report
        ---------------------------
        Workload: type={wtype}, num_ops={num_ops}, duration={duration}s, gap_min={gap_min}, gap_max={gap_max}, read_ratio={read_ratio}
        Disk: active_power={active_power}W, idle_power={idle_power}W, standby_power={standby_power}W
        spin_up_time={spin_up_time}s, spin_up_energy={spin_up_energy}J, spin_down_energy={spin_down_energy}J
        Predictor: {predictor_choice}, params={predictor_params}
        Idle threshold: {idle_threshold}s
        Aggressive factor (predictive): {aggressive_factor}

        Baseline energy (J): {base_energy:.2f}
        Predictive energy (J): {pred_energy:.2f}
        Energy saved (%): {saved_pct:.2f}
        """)
        st.download_button("Download Simulation Report (TXT)", data=report_text.encode(), file_name="san_report.txt", mime="text/plain")
