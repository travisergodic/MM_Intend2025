# install
cd /content/mm_intend
pip install --no-deps -e .
pip install -e ".[torch,metrics]"
pip install transformers==4.46.1
pip install datasets
pip install deepspeed==0.15.4