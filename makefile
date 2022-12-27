setup:
	git pull
	pip3 install -r requirements.txt
	pip3 install .

setup-dev:
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
	make train dataset=20News

evaluate-all:
	make evaluate dataset=MNIST && \
	make evaluate dataset=LfwPeople && \
	make evaluate dataset=SwissRoll && \
	make evaluate dataset=TwinPeaks && \
	make evaluate dataset=FER2013 && \
	make evaluate dataset=OlivettiFaces && \
	make evaluate dataset=20News

zip-results:
	zip -r models.zip models && \
	zip -r metrics.zip metrics


zip-model-for:
	zip -r $(dataset)-models.zip models/$(dataset)
