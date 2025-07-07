import re
import ast
import glob
import os
import matplotlib.pyplot as plt

# --- Models to plot
dict_models_all = {
    "model_6": 10,
    "model_10": 30,
    "model_0": 60,
    "model_9": 120
}

dict_models_img = {
    "model_2": 10,
    "model_7": 30,
    "model_1": 60,
    "model_8": 120
}
dict_model_3 = { # with only imgs
"model_3": 10,
"model_3": 30,
"model_3": 60,
}
dict_model_11 = { # with only imgs
"model_11": 10,
"model_11": 30,
"model_11": 60,
}

# --- Parsing helpers
def extract_np_float64(value):
    value = re.sub(r'np\.float64\(([^)]+)\)', r'\1', value)
    return float(value)

def extract_mae_values(text):
    result = {
        "global": {"geneva": None, "dole": None, "delta": None},
        "stratus": {"geneva": None, "dole": None, "delta": None}
    }

    # Global
    m_global = re.search(r"Mean Absolute Error:\s*\{[^}]*'geneva': ([^,}]+),\s*'dole': ([^}]+)\}", text)
    m_delta = re.search(r"=== Global Delta geneva-Dole Stats ===.*?'mae': (np\.float64\([^)]+\)|[0-9.]+)", text, re.DOTALL)
    if m_global:
        result["global"]["geneva"] = extract_np_float64(m_global.group(1))
        result["global"]["dole"] = extract_np_float64(m_global.group(2))
    if m_delta:
        result["global"]["delta"] = extract_np_float64(m_delta.group(1))

    # Stratus
    m_stratus_mae = re.search(r"=== Stratus Days Metrics ===.*?Global MAE: \{[^}]*'geneva': ([^,}]+),\s*'dole': ([^}]+)\}", text, re.DOTALL)
    m_stratus_delta = re.search(r"=== Stratus Days Metrics ===.*?Delta geneva-Dole Stats: \{[^}]*'mae': (np\.float64\([^)]+\)|[0-9.]+)", text, re.DOTALL)
    if m_stratus_mae:
        result["stratus"]["geneva"] = extract_np_float64(m_stratus_mae.group(1))
        result["stratus"]["dole"] = extract_np_float64(m_stratus_mae.group(2))
    if m_stratus_delta:
        result["stratus"]["delta"] = extract_np_float64(m_stratus_delta.group(1))

    return result

# --- Aggregator function
def collect_all_mae(model_dict):
    data = {
        "horizons": [],
        "delta_global": [],
        "delta_stratus": [],
        "geneva_global": [],
        "dole_global": [],
        "geneva_stratus": [],
        "dole_stratus": []
    }

    for model, horizon in sorted(model_dict.items(), key=lambda x: x[1]):
        pattern = f"models/{model}/metrics*/metrics_report.txt"
        print(pattern)
        matches = glob.glob(pattern)
        print(matches)
        if not matches:
            print(f"[!] Missing file for {model}")
            continue
        path = matches[0]
        with open(path, "r") as f:
            content = f.read()

        metrics = extract_mae_values(content)

        data["horizons"].append(horizon)
        data["delta_global"].append(metrics["global"]["delta"])
        data["delta_stratus"].append(metrics["stratus"]["delta"])
        data["geneva_global"].append(metrics["global"]["geneva"])
        data["dole_global"].append(metrics["global"]["dole"])
        data["geneva_stratus"].append(metrics["stratus"]["geneva"])
        data["dole_stratus"].append(metrics["stratus"]["dole"])
    print(data)

    return data

# --- Plot separate MAE
def plot_mae_components(data, title, filename):
    plt.figure(figsize=(9, 5))
    plt.plot(data["horizons"], data["geneva_global"], marker='o', label="Geneva Global", color="blue")
    plt.plot(data["horizons"], data["dole_global"], marker='s', label="Dole Global", color="green")
    plt.plot(data["horizons"], data["geneva_stratus"], marker='o', linestyle='--', label="Geneva Stratus", color="deepskyblue")
    plt.plot(data["horizons"], data["dole_stratus"], marker='s', linestyle='--', label="Dole Stratus", color="limegreen")

    plt.xlabel("Horizon (minutes)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved: {filename}")

# --- Run
data_all = collect_all_mae(dict_models_all)
data_img = collect_all_mae(dict_models_img)
def collect_all_mae_v2(model_dict):
    data = {
        "horizons": [],
        "delta_global": [],
        "delta_stratus": [],
        "geneva_global": [],
        "dole_global": [],
        "geneva_stratus": [],
        "dole_stratus": []
    }

    for model, horizons in model_dict.items():
        for horizon in sorted(horizons):
          
            pattern = f"models/{model}/metrics_{horizon}/metrics_report.txt"
            print(pattern)
            matches = glob.glob(pattern)
            print(matches)
            if not matches:
                print(f"[!] Missing file for {model} horizon {horizon}")
                continue
            path = matches[0]
            with open(path, "r") as f:
                content = f.read()

            metrics = extract_mae_values(content)

            data["horizons"].append(horizon)
            data["delta_global"].append(metrics["global"]["delta"])
            data["delta_stratus"].append(metrics["stratus"]["delta"])
            data["geneva_global"].append(metrics["global"]["geneva"])
            data["dole_global"].append(metrics["global"]["dole"])
            data["geneva_stratus"].append(metrics["stratus"]["geneva"])
            data["dole_stratus"].append(metrics["stratus"]["dole"])

    print(data)
    return data


dict_model_3 = {
    "model_3": ["t_0","t_1", "t_2", "t_3","t_4","t_5"]
}

dict_model_11 = {
    "model_11": ["t_0", "t_1", "t_2", "t_3","t_4","t_5"]
}

data_mul_out_img = collect_all_mae_v2(dict_model_3)
data_mul_out_img_11 = collect_all_mae_v2(dict_model_11)
# Map t_0, t_2, t_5 to 10, 20, 30 for horizons in data_mul_out_img and data_mul_out_img_11
def convert_t_to_minutes(horizons):
    mapping = {"t_0": 10, "t_1": 20, "t_2": 30, "t_3": 40, "t_4": 50, "t_5": 60}
    return [mapping.get(h, h) for h in horizons]

data_mul_out_img["horizons"] = convert_t_to_minutes(data_mul_out_img["horizons"])
data_mul_out_img_11["horizons"] = convert_t_to_minutes(data_mul_out_img_11["horizons"])


def plot_mae_and_delta_combined(data_all, data_img, data_mul_out_img, data_mul_out_img_11, title, filename):
    fig, axs = plt.subplots(3,1, figsize=(10, 14), sharex=True, gridspec_kw={"hspace": 0.3})
    # --- Subplot 1: MAE Stratus Geneva
    axs[0].plot(data_all["horizons"], data_all["geneva_stratus"], marker='o', label="Geneva Stratus (ALL)", color="#1f77b4")         # blue
    axs[0].plot(data_img["horizons"], data_img["geneva_stratus"], marker='o', linestyle='--', label="Geneva Stratus (IMG)", color="#e377c2")   # pink
    axs[0].plot(data_mul_out_img["horizons"], data_mul_out_img["geneva_stratus"], marker='o', linestyle=':', label="Geneva Stratus (IMG mult out)", color="#ff7f0e") # orange
    axs[0].plot(data_mul_out_img_11["horizons"], data_mul_out_img_11["geneva_stratus"], marker='o', linestyle='-.', label="Geneva Stratus (ALL mult out)", color="#2ca02c") # green
    axs[0].set_ylabel("MAE Stratus")
    axs[0].set_title(title + " – Geneva Stratus")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    # --- Subplot 2: MAE Stratus Dole
    axs[1].plot(data_all["horizons"], data_all["dole_stratus"], marker='s', label="Dole Stratus (ALL)", color="#d62728")            # red
    axs[1].plot(data_img["horizons"], data_img["dole_stratus"], marker='s', linestyle='--', label="Dole Stratus (IMG)", color="#9467bd")      # purple
    axs[1].plot(data_mul_out_img["horizons"], data_mul_out_img["dole_stratus"], marker='s', linestyle=':', label="Dole Stratus (IMG mult out)", color="#8c564b") # brown
    axs[1].plot(data_mul_out_img_11["horizons"], data_mul_out_img_11["dole_stratus"], marker='s', linestyle='-.', label="Dole Stratus (ALL mult out)", color="#17becf") # cyan
    axs[1].set_ylabel("MAE Stratus")
    axs[1].set_title(title + " – Dole Stratus")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    # --- Subplot 3: Delta MAE Stratus
    axs[2].plot(data_all["horizons"], data_all["delta_stratus"], marker='s', label="Delta Stratus (ALL)", color="#bcbd22")          # olive
    axs[2].plot(data_img["horizons"], data_img["delta_stratus"], marker='s', linestyle='--', label="Delta Stratus (IMG)", color="#7f7f7f")    # gray
    axs[2].plot(data_mul_out_img["horizons"], data_mul_out_img["delta_stratus"], marker='s', linestyle=':', label="Delta Stratus (IMG mult out)", color="#aec7e8") # light blue
    axs[2].plot(data_mul_out_img_11["horizons"], data_mul_out_img_11["delta_stratus"], marker='s', linestyle='-.', label="Delta Stratus (ALL mult out)", color="#ffbb78") # light orange
    axs[2].set_ylabel("Δ MAE (Geneva − Dole)")
    axs[2].set_title("Delta MAE Stratus")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved: {filename}")
    
plot_mae_and_delta_combined(data_all, data_img, data_mul_out_img, data_mul_out_img_11, "MAE + Δ MAE (ALL Models)", "mae_delta_combined_all.png")