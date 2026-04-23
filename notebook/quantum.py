# --------------------------------------------------------- #
#                                                           #
#                 THE ENTANGLED ANCILLAS                    #
#                                                           #
# --------------------------------------------------------- #

# --------------------------------------------------------- #
#                                                           #
#    A QUANTUM ALGORITHM for SIMPLE HARMONIC OSCILLATOR     #
#                                                           #
# --------------------------------------------------------- #


import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
from classiq import *
from classiq import ExecutionPreferences
from scipy.linalg import expm
from scipy.special import factorial

# classiq.authenticate()  # needed only once


# -------------------------------------------------- VARIABLES & CONSTANTS -------------------------------------------------- #
omega = 1
bound = 0.01

x0 = np.array([1.0, 1.0])
norm_x0 = np.linalg.norm(x0)
probs_x0 = ((x0 / norm_x0) ** 2).tolist()
amps_x0 = (x0 / norm_x0).tolist()

A = np.array([[0, 1], [-1, 0]])
norm_A = np.linalg.norm(A, ord=2)


# -------------------------------------------------- CIRCUIT & STATE OPERATIONS -------------------------------------------------- #
def compute_taylor_coeffs(t, k):
    # From paper: Cm = ||x(0)|| * (||A||*t)^m / m!
    Cm = [norm_x0 * (norm_A * t) ** m / factorial(m) for m in range(k + 1)]
    probabilities = [c / sum(Cm) for c in Cm]

    n_qubits_controller = math.ceil(math.log2(k + 1))
    n_states = 2**n_qubits_controller

    probabilities = probabilities + [0.0] * (n_states - len(probabilities))
    return Cm, probabilities


def build_circuit(bound, probs_controller, k, A_powers):
    n_qubits_controller = math.ceil(math.log2(k + 1))

    @qfunc
    def prepare(controller: QNum) -> None:
        inplace_prepare_state(probs_controller, bound=bound, target=controller)

    @qfunc
    def select(controller: QNum, work: QArray) -> None:
        for m in range(k + 1):
            control(
                ctrl=controller == m,
                stmt_block=lambda m=m: unitary(elements=A_powers[m], target=work),
            )

    @qfunc
    def main(controller: Output[QNum], work: Output[QArray]) -> None:
        allocate(n_qubits_controller, controller)
        allocate(1, work)  # 1 qubit for 2D inital state

        # Prepare initial state |x(0)>
        inplace_prepare_state(probabilities=probs_x0, bound=bound, target=work)

        # LCU: PREPARE† SELECT PREPARE (PREPARE† is taken care of in within_apply)
        within_apply(
            within=lambda: prepare(controller),
            apply=lambda: select(controller, work),
        )

    return main


def synthesize_and_run_circuit(bound, probabilities, k, A_powers, n_shots):
    circuit = build_circuit(bound, probabilities, k, A_powers)

    qmod = create_model(circuit)
    qmod = set_execution_preferences(qmod, ExecutionPreferences(n_shots=n_shots))
    qprog = synthesize(qmod)

    depth = qprog.transpiled_circuit.depth
    width = qprog.data.width

    job = execute(qprog)
    return job, depth, width


def get_state_and_errors(t, results):
    y_shots, v_shots = 0, 0

    for state, count in results.parsed_counts:
        if state[1]["controller"] == 0:
            if state[1]["work"] == [0]:
                y_shots += count[1]
            elif state[1]["work"] == [1]:
                v_shots += count[1]

    total_post_selected = y_shots + v_shots
    prob_y = y_shots / total_post_selected
    prob_v = v_shots / total_post_selected

    classical = expm(A * t) @ x0
    y_quantum = np.sign(classical[0]) * np.sqrt(prob_y) * norm_x0
    v_quantum = np.sign(classical[1]) * np.sqrt(prob_v) * norm_x0

    err_y = abs(y_quantum - classical[0])
    err_v = abs(v_quantum - classical[1])
    return {
        "classical": classical,
        "y_quantum": y_quantum,
        "v_quantum": v_quantum,
        "y_shots": y_shots,
        "v_shots": v_shots,
        "prob_y": prob_y,
        "prob_v": prob_v,
        "err_y": err_y,
        "err_v": err_v,
        "total_post_selected": total_post_selected,
    }


# -------------------------------------------------- ENERGY ANALYSIS -------------------------------------------------- #
positions_classical, velocities_classical = [], []
positions_quantum, velocities_quantum = [], []
kinetic_energy_classical, potential_energy_classical, total_energy_classical = (
    [],
    [],
    [],
)
kinetic_energy_quantum, potential_energy_quantum, total_energy_quantum = [], [], []
sigma_y_list, sigma_v_list = [], []
sigma_EK, sigma_EP, sigma_E = [], [], []

observed_rates = []
theoretical_rates = []


def plot_energy(times):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    # ── Position ──
    axes[0].plot(times, positions_classical, label="classical", color="blue")
    axes[0].errorbar(
        times,
        positions_quantum,
        yerr=sigma_y_list,
        label="quantum",
        color="red",
        fmt="o",
        capsize=4,
    )
    axes[0].set_title("y(t)")
    axes[0].set_xlabel("t")
    axes[0].legend()

    # ── Velocity ──
    axes[1].plot(times, velocities_classical, label="classical", color="blue")
    axes[1].errorbar(
        times,
        velocities_quantum,
        yerr=sigma_v_list,
        label="quantum",
        color="red",
        fmt="o",
        capsize=4,
    )
    axes[1].set_title("v(t)")
    axes[1].set_xlabel("t")
    axes[1].legend()

    # ── Energy ──
    axes[2].plot(times, total_energy_classical, label="E classical", color="blue")
    axes[2].errorbar(
        times,
        total_energy_quantum,
        yerr=sigma_E,
        label="E quantum",
        color="red",
        fmt="o",
        capsize=4,
    )
    axes[2].plot(
        times,
        kinetic_energy_classical,
        label="Ek classical",
        color="green",
        linestyle="--",
    )
    axes[2].plot(
        times,
        potential_energy_classical,
        label="Ep classical",
        color="orange",
        linestyle="--",
    )
    axes[2].errorbar(
        times,
        kinetic_energy_quantum,
        yerr=sigma_EK,
        label="Ek quantum",
        color="green",
        fmt="^",
        capsize=4,
    )
    axes[2].errorbar(
        times,
        potential_energy_quantum,
        yerr=sigma_EP,
        label="Ep quantum",
        color="orange",
        fmt="^",
        capsize=4,
    )
    axes[2].set_title("Energy")
    axes[2].set_xlabel("t")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"energy-analysis-{datetime.datetime.now()}.png")
    plt.show()


def plot_postselection_rate(times, observed_rates, theoretical_rates):
    plt.figure(figsize=(8, 4))
    plt.plot(
        times,
        theoretical_rates,
        label="theoretical $1/\\mathcal{N}^2$",
        color="blue",
        linestyle="--",
    )
    plt.plot(times, observed_rates, label="observed", color="red", marker="o")
    plt.xlabel("t")
    plt.ylabel("post-selection rate")
    plt.title("Post-Selection Rate: Observed vs Theoretical")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("postselection_rate.png", dpi=150)
    plt.show()


def calculate_energy(classical, y_quantum, v_quantum, sigma_y, sigma_v):
    y_classical, v_classical = classical

    # ── append positions and velocities ──
    positions_classical.append(y_classical)
    velocities_classical.append(v_classical)
    positions_quantum.append(y_quantum)
    velocities_quantum.append(v_quantum)

    # classical
    kinetic_energy_classical.append(0.5 * v_classical**2)
    potential_energy_classical.append(0.5 * omega**2 * y_classical**2)
    total_energy_classical.append(
        0.5 * v_classical**2 + 0.5 * omega**2 * y_classical**2
    )

    # quantum
    kinetic_energy_quantum.append(0.5 * v_quantum**2)
    potential_energy_quantum.append(0.5 * omega**2 * y_quantum**2)
    total_energy_quantum.append(0.5 * v_quantum**2 + 0.5 * omega**2 * y_quantum**2)

    # statistical errors:
    sigma_y_list.append(sigma_y)
    sigma_v_list.append(sigma_v)

    # propagate to energy uncertainty
    sig_EK = abs(v_quantum) * sigma_v
    sig_EP = abs(y_quantum) * sigma_y
    sig_E = np.sqrt(sig_EK**2 + sig_EP**2)

    sigma_EK.append(sig_EK)
    sigma_EP.append(sig_EP)
    sigma_E.append(sig_E)

    return


def analyse_energy_results(t, results, Cm):
    state = get_state_and_errors(t, results)

    classical = state["classical"]
    y_quantum = state["y_quantum"]
    v_quantum = state["v_quantum"]
    y_shots = state["y_shots"]
    v_shots = state["v_shots"]
    prob_y = state["prob_y"]
    prob_v = state["prob_v"]
    total_post_selected = state["total_post_selected"]

    theoretical_rate = 1 / sum(Cm) ** 2
    observed_rate = total_post_selected / results.num_shots
    theoretical_rates.append(theoretical_rate)
    observed_rates.append(observed_rate)

    # shot noise uncertainty
    sigma_prob_y = np.sqrt(prob_y * (1 - prob_y) / total_post_selected)
    sigma_prob_v = np.sqrt(prob_v * (1 - prob_v) / total_post_selected)
    sigma_y = (norm_x0 / (2 * np.sqrt(prob_y))) * sigma_prob_y
    sigma_v = (norm_x0 / (2 * np.sqrt(prob_v))) * sigma_prob_v

    print(f"{'─'*50}")
    print(f"  t = {t:.2f}")
    print(f"{'─'*50}")
    print(f"counts: {results.counts}")
    print(f"n_shots: {results.num_shots}")

    print()
    print(f"post-selected shots: {total_post_selected}")
    print(f"y shots: {y_shots} ({100*y_shots/total_post_selected:.1f}%)")
    print(f"v shots: {v_shots} ({100*v_shots/total_post_selected:.1f}%)")

    print(f"\nClassical solution at t={t:.2f}:")
    print(f"  y(t) = {classical[0]:.4f}")
    print(f"  v(t) = {classical[1]:.4f}")

    print(f"\nQuantum solution at t={t}:")
    print(f"  y(t) = {y_quantum:.4f}")
    print(f"  v(t) = {v_quantum:.4f}")

    print(f"\nError:")
    print(f"  y error = {abs(y_quantum - classical[0]):.4f}")
    print(f"  v error = {abs(v_quantum - classical[1]):.4f}")

    calculate_energy(classical, y_quantum, v_quantum, sigma_y, sigma_v)


def energy_analysis():

    k = 5
    bound = 0.01
    n_shots = 8192
    time_intervals_count = 21
    times = np.linspace(0, 1, time_intervals_count)

    A_powers = [np.linalg.matrix_power(A, m).tolist() for m in range(k + 1)]

    for time in times:
        Cm, probabilities = compute_taylor_coeffs(time, k)
        job, _, _ = synthesize_and_run_circuit(
            bound, probabilities, k, A_powers, n_shots
        )
        results = job.get_sample_result()
        analyse_energy_results(time, results, Cm)

    # plot_energy(times)
    plot_postselection_rate(times, observed_rates, theoretical_rates)


# energy_analysis()


# -------------------------------------------------- BOUND ANALYSIS -------------------------------------------------- #
def plot_bound(results_grid, time_points, bounds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effect of Bound Parameter on Error (k=5)", fontsize=13)

    for t_val in time_points:
        ey = [results_grid[(t_val, b)][0] for b in bounds]
        ev = [results_grid[(t_val, b)][1] for b in bounds]
        axes[0].plot(bounds, ey, "o-", label=f"t={t_val}")
        axes[1].plot(bounds, ev, "o-", label=f"t={t_val}")

    axes[0].set_title("Position Error vs Bound")
    axes[0].set_xlabel("bound")
    axes[0].set_ylabel("error_y")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Velocity Error vs Bound")
    axes[1].set_xlabel("bound")
    axes[1].set_ylabel("error_v")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bound_analysis.png", dpi=150)
    plt.show()


bound_results = {}


def bound_analysis():

    k = 5
    bounds = [0.001, 0.01, 0.05, 0.1]
    times = [0.0, 0.2, 0.5, 0.75, 1.0]
    n_shots = 8192
    A_powers = [np.linalg.matrix_power(A, m).tolist() for m in range(k + 1)]

    for time in times:
        for bound in bounds:

            _, probabilities = compute_taylor_coeffs(time, k)

            job, _, _ = synthesize_and_run_circuit(
                bound, probabilities, k, A_powers, n_shots
            )
            results = job.get_sample_result()

            state = get_state_and_errors(time, results)
            err_y = state["err_y"]
            err_v = state["err_v"]
            bound_results[(time, bound)] = (err_y, err_v)
            print(
                f"t={time} | bound={bound} | error_y={err_y:.4f} | error_v={err_v:.4f}"
            )

    plot_bound(bound_results, times, bounds)


# bound_analysis()


# -------------------------------------------------- K ANALYSIS -------------------------------------------------- #
def plot_k_accuracy(k_results, taylor_cutoffs, time_points):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effect of Taylor Order k on Error (bound=0.01)", fontsize=13)

    for t_val in time_points:
        ey = [k_results[(k, t_val)][0] for k in taylor_cutoffs]
        ev = [k_results[(k, t_val)][1] for k in taylor_cutoffs]
        axes[0].plot(taylor_cutoffs, ey, "o-", label=f"t={t_val}")
        axes[1].plot(taylor_cutoffs, ev, "o-", label=f"t={t_val}")

    axes[0].set_title("Position Error vs k")
    axes[0].set_xlabel("k (Taylor order)")
    axes[0].set_ylabel("error_y")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Velocity Error vs k")
    axes[1].set_xlabel("k (Taylor order)")
    axes[1].set_ylabel("error_v")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("k_accuracy.png", dpi=150)
    plt.show()


def plot_k_resources(k_results, taylor_cutoffs):
    # take t=0.5 as representative — depth/width don't depend on t
    depths = [k_results[(k, 0.5)][2] for k in taylor_cutoffs]
    widths = [k_results[(k, 0.5)][3] for k in taylor_cutoffs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Circuit Resources vs Taylor Order k (bound=0.01)", fontsize=13)

    axes[0].bar(taylor_cutoffs, depths, color="steelblue", alpha=0.8)
    axes[0].set_title("Circuit Depth vs k")
    axes[0].set_xlabel("k (Taylor order)")
    axes[0].set_ylabel("depth")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(taylor_cutoffs, widths, color="coral", alpha=0.8)
    axes[1].set_title("Circuit Width vs k")
    axes[1].set_xlabel("k (Taylor order)")
    axes[1].set_ylabel("width (qubits)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("k_resources.png", dpi=150)
    plt.show()


def k_analysis():
    bound = 0.01
    taylor_cutoffs = [1, 2, 3, 4, 5, 6, 7, 8]
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_shots = 8192

    # {(k, t_val): (error_y, error_v, depth, width)}
    k_results = {}

    for k in taylor_cutoffs:
        A_powers = [np.linalg.matrix_power(A, m).tolist() for m in range(k + 1)]
        for time in time_points:
            _, probabilities = compute_taylor_coeffs(time, k)
            job, depth, width = synthesize_and_run_circuit(
                bound, probabilities, k, A_powers, n_shots
            )

            results = job.get_sample_result()

            state = get_state_and_errors(time, results)
            err_y = state["err_y"]
            err_v = state["err_v"]
            k_results[(k, time)] = (err_y, err_v, depth, width)
            print(
                f"k={k} | t={time} | error_v={err_v:.4f} | depth={depth} | width={width}"
            )

    plot_k_accuracy(k_results, taylor_cutoffs, time_points)
    plot_k_resources(k_results, taylor_cutoffs)


# k_analysis()


# -------------------------------------------------- N SHOTS ANALYSIS -------------------------------------------------- #


def plot_shots_trajectories(trajectory_results, shot_counts, times):
    fig, axes = plt.subplots(2, len(shot_counts), figsize=(18, 8))
    fig.suptitle(
        "Quantum vs Classical Trajectory for Different n_shots (k=5, bound=0.01)",
        fontsize=13,
    )

    classical_y = [expm(A * t) @ x0 for t in times]
    y_classical = [s[0] for s in classical_y]
    v_classical = [s[1] for s in classical_y]

    for i, n_shots in enumerate(shot_counts):
        y_quantum = [trajectory_results[(t, n_shots)]["y_quantum"] for t in times]
        v_quantum = [trajectory_results[(t, n_shots)]["v_quantum"] for t in times]
        sigma_y = [trajectory_results[(t, n_shots)]["sigma_y"] for t in times]
        sigma_v = [trajectory_results[(t, n_shots)]["sigma_v"] for t in times]

        # y(t)
        axes[0][i].plot(times, y_classical, label="classical", color="blue")
        axes[0][i].errorbar(
            times,
            y_quantum,
            yerr=sigma_y,
            label="quantum",
            color="red",
            fmt="o",
            capsize=3,
            markersize=3,
        )
        axes[0][i].set_title(f"y(t) | n_shots={n_shots}")
        axes[0][i].set_xlabel("t")
        axes[0][i].legend(fontsize=7)
        axes[0][i].grid(True, alpha=0.3)

        # v(t)
        axes[1][i].plot(times, v_classical, label="classical", color="blue")
        axes[1][i].errorbar(
            times,
            v_quantum,
            yerr=sigma_v,
            label="quantum",
            color="red",
            fmt="o",
            capsize=3,
            markersize=3,
        )
        axes[1][i].set_title(f"v(t) | n_shots={n_shots}")
        axes[1][i].set_xlabel("t")
        axes[1][i].legend(fontsize=7)
        axes[1][i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("shots_trajectories.png", dpi=150)
    plt.show()


def n_shots_analysis():
    k = 5
    bound = 0.01
    shot_counts = [256, 1024, 4096, 8192]
    times = np.linspace(0, 1, 21)

    A_powers = [np.linalg.matrix_power(A, m).tolist() for m in range(k + 1)]
    trajectory_results = {}

    for n_shots in shot_counts:
        print(f"\n--- n_shots = {n_shots} ---")
        for time in times:
            _, probabilities = compute_taylor_coeffs(time, k)
            job, _, _ = synthesize_and_run_circuit(
                bound, probabilities, k, A_powers, n_shots
            )
            execution = job.get_sample_result()
            state = get_state_and_errors(time, execution)

            # compute sigma_y and sigma_v
            prob_y = state["prob_y"]
            prob_v = state["prob_v"]
            total = state["total_post_selected"]
            sigma_prob_y = np.sqrt(prob_y * (1 - prob_y) / total)
            sigma_prob_v = np.sqrt(prob_v * (1 - prob_v) / total)
            sigma_y = (norm_x0 / (2 * np.sqrt(prob_y))) * sigma_prob_y
            sigma_v = (norm_x0 / (2 * np.sqrt(prob_v))) * sigma_prob_v

            trajectory_results[(time, n_shots)] = {
                "y_quantum": state["y_quantum"],
                "v_quantum": state["v_quantum"],
                "sigma_y": sigma_y,
                "sigma_v": sigma_v,
            }
            print(
                f"t={time:.2f} | err_y={state['err_y']:.4f} | err_v={state['err_v']:.4f}"
            )

    plot_shots_trajectories(trajectory_results, shot_counts, times)


# n_shots_analysis()
