setup:
	python3 -m venv .venv
	source .venv/bin/activate

install:
	git pull
	pip3 install -r requirements.txt
	pip3 install .

install-dev:
	git pull
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt
	pip3 install -e .
	pre-commit install

train:
	drcomp -m evaluate=False dataset=$(dataset) reducer=AE,CAE,kPCA,LLE,ConvAE,PCA

evaluate:
	drcomp -m evaluate=True dataset=$(dataset) reducer=AE,CAE,kPCA,LLE,ConvAE,PCA use_pretrained=True

train-all:
	make train dataset=MNIST && \
	make train dataset=LfwPeople && \
	make train dataset=SwissRoll && \
	make train dataset=TwinPeaks && \
	make train dataset=FER2013 && \
	make train dataset=OlivettiFaces && \
	make train dataset=ICMR && \
	make train dataset=FashionMNIST

evaluate-all:
	make evaluate dataset=MNIST && \
	make evaluate dataset=LfwPeople && \
	make evaluate dataset=SwissRoll && \
	make evaluate dataset=TwinPeaks && \
	make evaluate dataset=FER2013 && \
	make evaluate dataset=OlivettiFaces && \
	make evaluate dataset=ICMR && \
	make evaluate dataset=FashionMNIST

zip-results:
	zip -r models.zip models && \
	zip -r metrics.zip metrics


zip-model-for:
	zip -r $(dataset)-models.zip models/$(dataset)


tune-kpca:
	drcomp -m evaluate=True reducer=kPCA dataset=MNIST,FER2013,SwissRoll,TwinPeaks,OlivettiFaces,ICMR,LfwPeople reducer.gamma=0.01,0.05,0.1,0.2,0.5,0.7,1,2

tune-autoencoder:
	drcomp -m evaluate=True reducer=AE dataset=MNIST,FER2013,SwissRoll,TwinPeaks,OlivettiFaces,ICMR,LfwPeople reducer.lr=0.5,0.1,0.05,0.01,0.005

tune-cae:
	drcomp -m evaluate=True reducer=CAE dataset=SwissRoll,TwinPeaks,OlivettiFaces,ICMR,LfwPeople "reducer.AutoEncoderClass.hidden_layer_dims=[],[32]" "reducer.AutoEncoderClass.encoder_act_fn._target_=nn.Sigmoid,nn.ReLU,nn.Tanh" reducer.contractive_lambda=0.0001,0.001 wandb.project=drcomp-v2
