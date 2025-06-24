# FeDist

FeDist is a new FL method that
exploits the knowledge distillation paradigm. In practice, clients share only their
logits outputs and the server aggregates this information to update the global model. This design makes no assumptions about the network architecture, hence
each client can use the network best suited to its environment, and is optimized for non-IID data distributions, ensuring stable and robust convergence.
A preliminary empirical evaluation of FeDist on four real-world datasets with different data distributions, show that our paradigm achieves good prediction
performance, always on par or better than the standard FedAvg approach.

---

## **Installation**

We recommend setting up a new Conda environment with **Python >= 3.9**.

### **1. Create a Conda environment**
```bash
conda create -n "fedist" python==3.9
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
In this section, we provide details about the experimental setting.

### **Neural Networks**

Across all datasets, we train a feedforward neural network with two hidden layers containing $300$ and $100$ neurons, respectively, and ReLU activations. A Dropout layer with a rate of $0.2$ is applied after the last hidden layer for regularization, while the output layer uses the Softmax function.

### **Hyper-parameters**

Regarding training, we train the neural network for a maximum of $100$ global iterations at the server level and up to $5$ epochs per client. Early stopping is applied with a patience of $5$ epochs at the server level, based on the global $F_1$ score. At the client level, we employ the Adam optimizer with a learning rate of $1e^{-4}$, a weight decay factor of $1e^{-4}$, and a batch size of $128$. The loss function is the Cross-Entropy. We execute at most $5$ FedAvg epochs in the aggregation phase, with an early stopping condition (patience = 2).

---

## **Usage**

To run an experiment with `FeDist`:

### **1. Navigate to the `src` directory**
```bash
cd src
```

### **2. Execute `main.py` with options**
```bash
python main.py --options
```

### **Available Options**
```bash
Options:
  -r, --run TEXT                  Name of the run to execute
  -p, --project_name TEXT         Name of the WandB project
  -nl --num_local_iterations default=30,  Number of local epochs
  -nf --num_federated_iterations default=100,  Number of global epochs
  -nc --num_clients default=10,     Number of clients
  -e ---experiment_name  default='New'   Name of the experiment, folder containing the data
```

#### **Predefined Runs (`runs` folder)**
- **`folk_fedist`** → Uses the **Income** dataset.
- **`insurance_fedist`** → Uses the **Insurance** dataset.
- **`employment_fedist`** → Uses the **Employment** dataset.
- **`meps_fedist`** → Uses the **MEPS** dataset.