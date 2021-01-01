.PHONY: probe test clean glue

all: probe

probe:
	echo "Probing"
	./probe.sh < params

glue:
	echo "Fine tuning LM on GLUE benchmark"
	./GLUE/glue.sh < ./GLUE/params

test:
	echo "Testing unpruned pretrained LMs"
	python -W ignore -u utils.py albert-large-v2 3

clean:
	rm -rf *.log