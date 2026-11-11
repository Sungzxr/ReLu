# relu_app.py
# Streamlit app 1 of 3: Rectified Linear Unit (ReLU) activation function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ReLU Activation Function", layout="wide")
st.title("Rectified Linear Unit (ReLU)")
st.write(
    "ReLU passes positive values unchanged and maps negative values to zero. "
    "It adds non-linearity to a neuron while staying cheap to compute."
)
st.latex(r"f(z) = \max(0, z)")


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


# --- Sidebar controls ---
with st.sidebar:
    st.header("Neuron settings")
    x = st.slider("Input x", -5.0, 5.0, 1.5, 0.1)
    w = st.slider("Weight w", -3.0, 3.0, 1.0, 0.1)
    b = st.slider("Bias b", -3.0, 3.0, 0.0, 0.1)

    st.header("Plot settings")
    z_min, z_max = st.slider("z range", -10.0, 10.0, (-6.0, 6.0), 0.5)
    show_grad = st.checkbox("Show derivative", value=True)

# --- One neuron: z = w*x + b, output = ReLU(z) ---
z = w * x + b
out = relu(np.array(z))
grad = relu_grad(np.array(z))

st.subheader("1. One neuron calculation")
c1, c2, c3 = st.columns(3)
c1.metric("z = w·x + b", f"{z:.3f}")
c2.metric("ReLU(z)", f"{float(out):.3f}")
c3.metric("Derivative at z", f"{float(grad):.0f}")
st.write(f"z = ({w:.1f} × {x:.1f}) + {b:.1f} = **{z:.3f}**  →  ReLU(z) = **{float(out):.3f}**")

# --- Plot ---
st.subheader("2. Activation curve")
zs = np.linspace(z_min, z_max, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(zs, relu(zs), label="ReLU(z)", linewidth=2)
if show_grad:
    ax.plot(zs, relu_grad(zs), "--", label="Derivative", linewidth=1.5)
ax.scatter([z], [float(out)], color="red", zorder=5, label="Current neuron")
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("z (weighted sum)")
ax.set_ylabel("Output")
ax.set_title("ReLU activation function")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Value table ---
st.subheader("3. Sample values")
sample = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
st.dataframe(
    pd.DataFrame({"z": sample, "ReLU(z)": relu(sample), "Derivative": relu_grad(sample)}),
    hide_index=True,
)

# --- Notes ---
st.subheader("4. Key properties")
st.markdown(
    """
- **Output range:** 0 to +∞
- **Advantages:** very fast to compute, does not saturate for positive inputs, helps reduce vanishing gradients, produces sparse activations.
- **Disadvantages:** "dying ReLU" problem, where neurons stuck at z < 0 output 0 with zero gradient and stop learning; output is not zero-centred.
- **Typical use:** hidden layers of deep networks and CNNs.
"""
)
