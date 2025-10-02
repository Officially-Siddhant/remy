# pip install onnxruntime
import onnxruntime as ort
import numpy as np

# load the session
session = ort.InferenceSession("./models/tinyphysics.onnx")

# inspect inputs/outputs
print("Inputs:", [i.name for i in session.get_inputs()])
print("Outputs:", [o.name for o in session.get_outputs()])

# dummy test run
dummy_input = np.zeros(session.get_inputs()[0].shape, dtype=np.float32)
out = session.run(None, {session.get_inputs()[0].name: dummy_input})
print("Output sample:", out)