# 🔋 SAN Energy Simulator  
### Predictive vs Baseline Disk Power Management using Streamlit

An interactive simulation tool that analyzes Storage Area Network (SAN) disk energy consumption under baseline and predictive power management strategies.

This project demonstrates workload forecasting, disk state-machine modeling, and energy optimization using Python and Streamlit.

---

## 🚀 Project Overview

The SAN Energy Simulator compares two disk power management policies:

- **Baseline Policy** – Spins down disk after fixed idle threshold  
- **Predictive Policy** – Uses workload forecasting (Moving Average / EWMA) to spin down earlier  

The simulator measures:

- Total energy consumption (Joules)
- Disk state transitions (ACTIVE / IDLE / STANDBY)
- Spin-up and spin-down counts
- Request latency
- Energy savings percentage

---

## 🧠 Key Features

- Interactive Streamlit-based GUI
- Custom workload generation (Random / Sequential / Bursty)
- Moving Average predictor
- EWMA (Exponentially Weighted Moving Average) predictor
- Disk power-state simulation (ACTIVE, IDLE, STANDBY)
- Energy accounting with spin-up and spin-down modeling
- Data visualization using Matplotlib
- Exportable CSV and simulation report

---

## 🛠 Tech Stack

- Python 3
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

## 📂 Project Structure

```
san-energy-simulator/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── screenshots/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/varshh219/san-energy-simulator
cd san-energy-simulator
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

If `streamlit` is not recognized:

```bash
py -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📊 Simulation Logic

The disk is modeled using a state-machine approach:

### Disk States

- **ACTIVE** → Serving requests  
- **IDLE** → Spinning but not serving  
- **STANDBY** → Powered down  

Energy calculation:

```
Energy = Power × Time + Spin Transition Energy
```

Predictive models estimate future inter-arrival times to aggressively spin down during idle periods.

---

## 📈 Results

Under bursty workloads, the predictive policy achieves:

- Up to 25–30% energy savings
- Reduced idle power consumption
- Efficient spin-down decisions
- Controlled latency trade-offs

---

## 📤 Export Options

The application allows downloading:

- Request logs (CSV)
- Policy comparison summary (CSV)
- Simulation report (TXT)

---

## 📌 Future Improvements

- Multi-disk simulation support
- Real-world workload trace integration
- Advanced ML-based predictors
- Cloud deployment support

---

## 📄 License

This project is developed for academic and learning purposes.

---

⭐ If you found this project useful, feel free to star the repository.
