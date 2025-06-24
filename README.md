# FeDist

**FeDist** is a novel Federated Learning (FL) method based on the knowledge distillation paradigm. Clients share only their **logit outputs**, which the server aggregates to update the global model. This design is **architecture-agnostic**, allowing each client to use the network best suited to its environment. FeDist is optimized for **non-IID data distributions**, ensuring stable and robust convergence.

A preliminary empirical evaluation on four real-world datasets with varying data distributions demonstrates that **FeDist consistently achieves strong predictive performance**, matching or outperforming the standard **FedAvg** approach.

---

## **Installation**

We recommend setting up a new Conda environment with **Python ≥ 3.9**.

### **1. Create a Conda environment**
```bash
conda create -n "fedist" python=3.9
```

### **2. Activate the environment**
```bash
conda activate fedist
```

### **3. Clone the repository**
```bash
git clone https://anonymous.4open.science/r/WAFL_Workshop_2025-8255/
```

### **4. Navigate to the project directory**
```bash
cd WAFL_Workshop_2025
```

### **5. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Experimental Setting**

### **Neural Network Architecture**

For all datasets, we use a **feedforward neural network** with:

- Two hidden layers of **300** and **100** neurons
- **ReLU** activations
- A **Dropout** layer (rate = 0.2) after the last hidden layer for regularization
- A final **Softmax** output layer

### **Hyperparameters**

- **Global training**: Up to **100 global iterations**
- **Client-side training**: Up to **5 epochs per round**
- **Early stopping** at the server (patience = 5), based on the global **F₁ score**
- **Optimizer**: Adam
- **Learning rate**: `1e-4`
- **Weight decay**: `1e-4`
- **Batch size**: 128
- **Loss function**: Cross-Entropy
- **Aggregation phase**: Up to 5 **FedAvg** steps, with early stopping (patience = 2)

---

## **Usage**

To run an experiment with **FeDist**:

### **1. Navigate to the `src` directory**
```bash
cd src
```

### **2. Run `main.py` with the desired options**
```bash
python main.py --options
```

### **Available Command-Line Options**
```bash
Options:
  -r, --run TEXT                      Name of the run to execute
  -p, --project_name TEXT             Name of the WandB project
  -nl, --num_local_iterations INT     Number of local training epochs (default: 30)
  -nf, --num_federated_iterations INT Number of global training epochs (default: 100)
  -nc, --num_clients INT              Number of clients (default: 10)
  -e, --experiment_name TEXT          Name of the experiment (default: 'New')
```

### **Predefined Runs (`runs` folder)**

- **`folk_fedist`** → Uses the **Income** dataset  
- **`insurance_fedist`** → Uses the **Insurance** dataset  
- **`employment_fedist`** → Uses the **Employment** dataset  
- **`meps_fedist`** → Uses the **MEPS** dataset  

---
