# ML-Based Page Replacement System

An intelligent **machine learning–driven page replacement simulator** that predicts future memory accesses to reduce page faults in virtual memory management. This project compares ML-based predictions with traditional algorithms such as **LRU** and **FIFO**.

---

## 🚀 Features

* Predicts optimal page eviction using machine learning
* Simulates memory access traces
* Compares performance with LRU and FIFO algorithms
* Measures hit ratio and page fault rates
* Visualizes performance using graphs

---

## 🛠 Tech Stack

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**

---

## 📂 Project Structure

```
ml-page-replacement/
│
├── data/               # Memory trace datasets
├── models/             # Trained ML models
├── src/
│   ├── simulator.py    # Page replacement simulator
│   ├── ml_model.py     # ML training and prediction
│   └── utils.py        # Helper functions
│
├── results/            # Performance outputs and graphs
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/prashant1039/ml_page_replacement.git
cd ml_page_replacement
```

2. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## ▶️ Usage

Run the simulator:

```bash
python src/simulator.py
```

The program will:

* Train the ML model
* Simulate memory access patterns
* Compare results with LRU and FIFO
* Display performance graphs

---

## 📊 Results

The ML-based approach improves prediction of future page accesses and demonstrates:

* Reduced page faults
* Higher hit ratio
* Better adaptability to access patterns

---

## 🎯 Learning Outcomes

* Understanding of virtual memory management
* Application of machine learning to systems problems
* Performance benchmarking and evaluation
* Data preprocessing and model training

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## 📜 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Prashant Singh**

* GitHub: https://github.com/prashant1039
* LinkedIn: https://www.linkedin.com/in/prashant-singh-17588728a/
