import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def run_anomaly_detector():
    print("--- Soliton Industrial Monitor Initializing ---")
    
    # 1. Create a fake sensor signal (like a motor vibrating)
    t = np.linspace(0, 10, 500)
    # This creates a smooth wave + some normal background noise
    signal = np.sin(t) + np.random.normal(0, 0.1, 500)

    # 2. Ask you how many "failures" to create
    try:
        n = int(input("How many machine faults (anomalies) should I simulate? (Type a number like 5): "))
    except:
        n = 5

    # 3. Randomly place the faults in the signal
    fault_indices = np.random.randint(0, 500, n)
    signal[fault_indices] = signal[fault_indices] * 4  # Make the signal 4x bigger at fault points

    # 4. Use AI (Isolation Forest) to find those faults automatically
    model = IsolationForest(contamination=0.02)
    predictions = model.fit_predict(signal.reshape(-1, 1))
    # 5. Create the visual graph
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Plot the normal signal in green
    plt.plot(t, signal, color='lime', label='Normal Operation', alpha=0.5)
    
    # Plot the detected faults in red
    plt.scatter(t[predictions == -1], signal[predictions == -1], color='red', label='FAULT DETECTED', s=50)
    
    plt.title("Soliton Project 1: Real-time Anomaly Detection")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Sensor Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    print("Displaying Graph... Close the graph window to finish.")
    plt.show()

if __name__ == "__main__":
    run_anomaly_detector()