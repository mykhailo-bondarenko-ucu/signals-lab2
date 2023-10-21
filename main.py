from scipy.signal import lfilter, hilbert, unit_impulse, TransferFunction, dimpulse, convolve, freqz
from matplotlib import pyplot as plt
from scipy.io import wavfile
import soundfile as sf
import numpy as np

def difference_equation_from_date(date, print_diff_formula=False):
    day, month, year = date.split('.')
    m1, m2 = map(int, month)
    d1, d2 = map(int, day)
    p2, p3, p4 = map(int, year[1:])
    if print_diff_formula:
        print(f"y[n] + \\frac{{({d1} + {d2})}}{{140}} y[n-1] + \\frac{{({p2} - {d2})}}{{130}} y[n-2] - \\frac{{{d1}}}{{150}} y[n-4] - \\frac{{{m1} - {d1}}}{{150}} y[n-5] = \\frac{{{m1}}}{{10}} x[n] + \\frac{{({p3} - {d2})}}{{20}} x[n-1] - \\frac{{({m2} - {m1})}}{{20}} x[n-2] - \\frac{{{p4}}}{{30}} x[n-3] + \\frac{{{d2}}}{{20}} x[n-4] - \\frac{{{m2}}}{{20}} x[n-5]")

    a = np.array([
        1, (d1 + d2) / 140, (p2 - d2) / 130, 0, -d1 / 150, -(m1 - d1) / 150
    ])

    b = np.array([
        m1 / 10, (p3 - d2) / 20, -(m2 - m1) / 20, -p4 / 30, d2 / 20, -m2 / 20
    ])

    return a, b

def task_1():
    a, b = difference_equation_from_date("26.01.2003", print_diff_formula=True)
    print(f"{a = }")
    print(f"{b = }")

def add_signal_length_to_title(title, time_values_s: np.ndarray):
    dur_s = time_values_s.max() - time_values_s.min()
    plt.title(f"{title} ({dur_s:.2f} seconds)")

def create_figure(title, x: np.ndarray, xlab='', ylab='', figure=True, add_length=True):
    if figure: plt.figure(figsize=(8, 6))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    if x is not None:
        if add_length: add_signal_length_to_title(title, x)
        plt.xlim(x.min(), x.max())
    else:
        plt.title(title)

def gen_sin_signal(frequency_hz, amplitude, duration_s, sample_rate):
    n = np.arange(0, duration_s * sample_rate)
    time_values_s = n * (1/sample_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency_hz * time_values_s)
    return time_values_s, signal

def plot_signal_in_out(title, time_values_s, input_signal, output_signal, input_envelope=None, output_envelope=None):
    create_figure(title, time_values_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_values_s, input_signal, label="Input", linestyle='dashed', marker='o', markersize=4.3)
    plt.plot(time_values_s, output_signal, label="Output", marker='o', markersize=4.3)
    if input_envelope: plt.plot(time_values_s, input_envelope, 'b-', label="Input Envelope")
    if output_envelope: plt.plot(time_values_s, output_envelope, 'r-', label="Output Envelope")
    plt.legend()

def task_2():
    frequency_hz, amplitude_volts, duration_s, sample_rate  = 10, 1, 1, 256
    time_values_s, signal = gen_sin_signal(frequency_hz, amplitude_volts, duration_s, sample_rate)
    create_figure("Sin wave", time_values_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_values_s, signal)
    plt.savefig("task_2_sin.png")
    plt.show()

    a, b = difference_equation_from_date("26.01.2003")
    input_envelope = np.abs(hilbert(signal))

    output_zero_ic, _ = lfilter(
        b, a, signal,
        zi=np.zeros(max(len(a), len(b)) - 1)
    )
    output_envelope_zero = np.abs(hilbert(output_zero_ic))
    plot_signal_in_out("Response with Zero Initial Conditions", time_values_s, signal, output_zero_ic, input_envelope, output_envelope_zero)
    plt.savefig("task_2_zero_bd_response.png")
    plt.show()

    output_random_ic, _ = lfilter(
        b, a, signal,
        zi=np.random.rand(max(len(a), len(b)) - 1)
    )
    output_envelope_rand = np.abs(hilbert(output_random_ic))
    plot_signal_in_out("Response with Random Initial Conditions", time_values_s, signal, output_random_ic, input_envelope, output_envelope_rand)
    plt.savefig("task_2_random_bd_response.png")
    plt.show()

    time_values_100ms = time_values_s[time_values_s <= 0.1]
    input_envelope_100ms = input_envelope[:len(time_values_100ms)]
    signal_100ms = signal[:len(time_values_100ms)]

    output_zero_ic_100ms = output_zero_ic[:len(time_values_100ms)]
    output_envelope_zero_100ms = output_envelope_zero[:len(time_values_100ms)]
    plot_signal_in_out("Response (First 100ms) with Zero Initial Conditions", time_values_100ms, signal_100ms, output_zero_ic_100ms, input_envelope_100ms, output_envelope_zero_100ms)
    plt.savefig("task_2_zero_bd_response_100ms.png")
    plt.show()

    output_random_ic_100ms = output_random_ic[:len(time_values_100ms)]
    output_envelope_rand_100ms = output_envelope_rand[:len(time_values_100ms)]
    plot_signal_in_out("Response (First 100ms) with Random Initial Conditions", time_values_100ms, signal_100ms, output_random_ic_100ms, input_envelope_100ms, output_envelope_rand_100ms)
    plt.savefig("task_2_random_bd_response_100ms.png")
    plt.show()

def H_omega(b, a, omega):
    """ H(e^{j * omega}) """
    num = b @ np.exp(-1j * np.arange(len(b)) * omega)
    denom = a @ np.exp(-1j * np.arange(len(a)) * omega)
    return num / denom

def task_3():
    a, b = difference_equation_from_date("26.01.2003")
    frequency_hz, amplitude_volts, duration_s, sample_rate  = 10, 1, 1, 256
    time_values_s, input_signal = gen_sin_signal(frequency_hz, amplitude_volts, duration_s, sample_rate)

    # frequency should be in rad/sample, not hz.
    # if freq rotations in a second, then 2pi*freq is the angle of rotation
    # 2pi*freq / sample_rate is the angle of rotation per sample
    omega = 2 * np.pi * frequency_hz / sample_rate
    H_10hz = H_omega(b, a, omega)
    print(f"Gain at 10Hz: {np.abs(H_10hz)}")
    print(f"Phase shift at 10 Hz (rads in period): {np.angle(H_10hz)}")
    shift_seconds = np.angle(H_10hz) / (2 * np.pi * frequency_hz)
    print(f"Phase shift at 10 Hz (seconds): {shift_seconds}")

    output_zero_ic, _ = lfilter(b, a, input_signal, zi=np.zeros(max(len(a), len(b)) - 1))
    create_figure("Response with Zero Initial Conditions. Real and predicted.", time_values_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_values_s, input_signal, label="Input", linestyle='dashed', marker='o', markersize=4.3)
    plt.plot(time_values_s, output_zero_ic, label="Output", marker='o', markersize=4.3)
    plt.plot(time_values_s - shift_seconds, input_signal * np.abs(H_10hz), 'r-', label="Predicted")
    plt.legend()
    plt.savefig("task_3.png")
    plt.show()


def task_4():
    duration_s, sample_rate  = 1, 256

    time_signal_s, signal_3hz = gen_sin_signal(3, 1, duration_s, sample_rate)
    _, signal_20hz = gen_sin_signal(20, 1, duration_s, sample_rate)

    create_figure("Original Signals", time_signal_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_signal_s, signal_3hz, label="3Hz Signal")
    plt.plot(time_signal_s, signal_20hz, label="20Hz Signal")
    plt.legend()
    plt.savefig("task_4_original_signals.png")
    plt.show()

    a, b = difference_equation_from_date("26.01.2003")

    response_3hz = lfilter(b, a, signal_3hz)
    response_20hz = lfilter(b, a, signal_20hz)
    signal_sum = signal_3hz + signal_20hz
    response_sum = lfilter(b, a, signal_sum)

    scaled_signal_3hz = 2 * signal_3hz
    response_scaled_3hz = lfilter(b, a, scaled_signal_3hz)

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_signal_s, response_3hz, label="Response to 3Hz Signal")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude (V)"); plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_signal_s, response_20hz, label="Response to 20Hz Signal")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude (V)"); plt.legend(); plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_signal_s, response_sum, label="Response to Summed Signals (3Hz + 20Hz)")
    plt.plot(time_signal_s, response_3hz + response_20hz, 'r--', label="Sum of Responses")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude (V)"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("task_4_responses.png")
    plt.show()

    create_figure("Response Comparisons", time_signal_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_signal_s, response_3hz, label="Response to 3Hz Signal")
    plt.plot(time_signal_s, response_scaled_3hz, label="Response to Scaled 3Hz Signal")
    plt.plot(time_signal_s, response_3hz * 2, 'r--', label="Scaled Response")
    plt.legend()
    plt.savefig("task_4_homogeneity.png")
    plt.show()


def task_5():
    sample_rate = 256
    a, b = difference_equation_from_date("26.01.2003")
    impulse_signal = unit_impulse(30)
    impulse_response = lfilter(b, a, impulse_signal)
    time_values_s = np.arange(30) * (1 / sample_rate)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Unit Impulse"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude (V)"); plt.grid(True)
    plt.stem(time_values_s, impulse_signal, use_line_collection=True)

    plt.subplot(2, 1, 2)
    plt.title("Impulse Response"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude (V)"); plt.grid(True)
    plt.stem(time_values_s, impulse_response, use_line_collection=True)

    plt.tight_layout()
    plt.savefig("task_5_impulse_response.png")
    plt.show()


def task_6():
    sample_rate = 256
    a, b = difference_equation_from_date("26.01.2003")

    # Raises a warning because of leading zeroes in b
    # We choose to ignore it.
    tf = TransferFunction(b, a, dt=1/sample_rate)
    t, y = dimpulse(tf, n=30)
    impulse_response_tf = y[0].flatten()

    plt.figure(figsize=(10, 6))
    impulse_signal = unit_impulse(30)
    impulse_response_lfilter = lfilter(b, a, impulse_signal)
    time_values_s = np.arange(30) * (1 / sample_rate)

    print(f"{np.allclose(impulse_response_tf, impulse_response_lfilter) = }")

    plt.subplot(2, 1, 1)
    create_figure("Impulse Response using lfilter", time_values_s, "Time (s)", "Amplitude (V)", figure=False)
    plt.stem(time_values_s, impulse_response_lfilter, use_line_collection=True)

    plt.subplot(2, 1, 2)
    create_figure("Impulse Response using TransferFunction", time_values_s, "Time (s)", "Amplitude (V)", figure=False)
    plt.stem(time_values_s, impulse_response_tf, use_line_collection=True, linefmt='r-', markerfmt='ro')

    plt.tight_layout()
    plt.savefig("task_6_compare_impulse_responses.png")
    plt.show()

    impulse_response_tf_100 = dimpulse(tf, n=100)[1][0].flatten()
    impulse_response_lfilter_100 = lfilter(b, a, unit_impulse(100))

    print(f"{np.allclose(impulse_response_tf_100, impulse_response_lfilter_100) = }")


def task_7():
    frequency_hz, amplitude_volts, duration_s, sample_rate  = 10, 1, 1, 256
    time_values_s, signal = gen_sin_signal(frequency_hz, amplitude_volts, duration_s, sample_rate)
    a, b = difference_equation_from_date("26.01.2003")
    output_lfilter, _ = lfilter(b, a, signal, zi=np.zeros(max(len(a), len(b)) - 1))

    h = lfilter(b, a, unit_impulse(1000))
    output_convolve = convolve(signal, h)[:len(signal)]

    print(f"{np.allclose(output_lfilter, output_convolve) = }")

    plot_signal_in_out("Response with Zero IC, convolve()", time_values_s, signal, output_convolve)
    plt.savefig("task_7_zero_bd_response.png")
    plt.show()


def task_8():
    sample_rate = 256
    a, b = difference_equation_from_date("26.01.2003")

    w_hz, h = freqz(b, a, worN=100, fs=sample_rate)

    amplitude_response = np.abs(h)
    phase_response = np.angle(h)

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    create_figure("Amplitude Response", w_hz, "Frequency (Hz)", "Voltage gain", figure=False, add_length=False)
    plt.plot(w_hz, amplitude_response, 'r')

    plt.subplot(2, 1, 2)
    create_figure("Phase Response", w_hz, "Frequency (Hz)", "Phase (radians in period)", figure=False, add_length=False)
    plt.plot(w_hz, phase_response, 'b')
    
    plt.savefig("task_8_amplitude_and_phase_Response.png")
    plt.show()


def task_9():
    sample_rate = 256
    a, b = difference_equation_from_date("26.01.2003")
    w_hz, h = freqz(b, a, worN=100, fs=sample_rate)
    amplitude_response = np.abs(h)

    gain_greater_than_one = np.where(amplitude_response > 1)[0]

    if len(gain_greater_than_one) == 0:
        print("There are no intervals with gain > 1")
    else:
        intervals = []
        start = gain_greater_than_one[0]
        for i in range(1, len(gain_greater_than_one)):
            if gain_greater_than_one[i] != gain_greater_than_one[i-1] + 1:
                intervals.append((start, gain_greater_than_one[i-1]))
                start = gain_greater_than_one[i]
        intervals.append((start, gain_greater_than_one[-1]))

        print("Intervals with gain > 1:")
        for interval in intervals:
            print(f"[{w_hz[interval[0]]:3.2f}, {w_hz[interval[1]]:3.2f}] Hz,")


def get_gain_and_shift_in_s(b, a, frequency_hz, sample_rate):
    omega = 2 * np.pi * frequency_hz / sample_rate
    H_o = H_omega(b, a, omega)
    return np.abs(H_o), np.angle(H_o) / (2 * np.pi * frequency_hz)

def freqz_get_gain_and_shift_in_s(b, a, frequency_hz, sample_rate):
    w_hz, h = freqz(b, a, worN=100, fs=sample_rate)
    H_o = h[np.argmin(np.abs(w_hz - frequency_hz))]
    return np.abs(H_o), np.angle(H_o) / (2 * np.pi * frequency_hz)

def task_10():
    frequency_hz, amplitude_volts, duration_s, sample_rate = 10, 1, 1, 256
    time_values_s, signal = gen_sin_signal(frequency_hz, amplitude_volts, duration_s, sample_rate)
    a, b = difference_equation_from_date("26.01.2003")
    output_lfilter, _ = lfilter(b, a, signal, zi=np.zeros(max(len(a), len(b)) - 1))

    g, an_s = get_gain_and_shift_in_s(b, a, frequency_hz, sample_rate)
    fz_g, fz_an_s = freqz_get_gain_and_shift_in_s(b, a, frequency_hz, sample_rate)

    print(f"Using H formula: {g = :1.9},    {an_s = :2.9}")
    print(f"Using freqz:  {fz_g = :1.9}, {fz_an_s = :2.9}")

    output_predicted = g * np.sin(2 * np.pi * frequency_hz * (time_values_s + an_s))

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    create_figure("Response with Zero Initial Conditions. Real and predicted.", time_values_s[:30], "Time (s)", "Amplitude (V)", figure=False)
    plt.plot(time_values_s[:30], signal[:30], label="Input", linestyle='dashed', marker='o', markersize=4.3)
    plt.plot(time_values_s[:30], output_lfilter[:30], label="Output", marker='o', markersize=4.3)
    plt.plot(time_values_s[:30], output_predicted[:30], 'r-', label="Predicted")
    plt.legend()

    plt.subplot(1, 2, 2)
    create_figure("Absolute distance between predicted and actual", time_values_s[:30], 'Time (s)', 'Absolute distance (V)', figure=False)
    plt.plot(time_values_s, np.abs(output_predicted - output_lfilter))

    plt.savefig("task_10.png")
    plt.show()

    print(f"{np.allclose(output_predicted[16:], output_lfilter[16:]) = }")


def generate_rect_pulse_sequence(duration_s, sample_rate, duty_cycle):
    num_samples = int(duration_s * sample_rate)
    pulse_width = int(duty_cycle * sample_rate)  # in samples
    pulse_sequence = np.array([i % sample_rate < pulse_width for i in range(num_samples)], dtype=np.int64)
    time_values_s = np.arange(0, duration_s, 1/sample_rate)
    return time_values_s, pulse_sequence

def task_11():
    duration_s, sample_rate = 10, 256
    a, b = difference_equation_from_date("26.01.2003")
    time_values_s, rect_pulse = generate_rect_pulse_sequence(duration_s, sample_rate, 0.3)
    output, _ = lfilter(b, a, rect_pulse, zi=np.zeros(max(len(a), len(b)) - 1))

    create_figure("Rectangular Pulse Sequence and LDS Response", time_values_s, "Time (s)", "Amplitude (V)")
    plt.plot(time_values_s, rect_pulse, 'b', label="Rectangular Pulse")
    plt.plot(time_values_s, output, 'r', label="LDS Response")
    plt.legend()
    plt.savefig("task_11.png")
    plt.show()

def process_and_save(filename, a, b, out_filename):
    data, sample_rate = sf.read(filename)
    output = lfilter(b, a, data)
    wavfile.write(out_filename, sample_rate, output)

def task_13():
    a, b = difference_equation_from_date("26.01.2003")
    process_and_save("audio_8k.wav", a, b, "out_audio_8k.wav")
    process_and_save("audio_44k.wav", a, b, "out_audio_44k.wav")

def main():
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()
    task_6()
    task_7()
    task_8()
    task_9()
    task_10()
    task_11()
    task_13()


if __name__ == "__main__":
    main()
