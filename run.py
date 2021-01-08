
from streamlit import bootstrap

real_script = "honey-bee-health-detection-with-cnn.py"

bootstrap.run(real_script, f'run.py {real_script}', [])