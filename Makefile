.PHONY: probe test clean glue

probe:
	./probe.sh < params

glue:
	./GLUE/glue.sh < ./GLUE/params

test:
	echo "Testing unpruned pretrained LMs"
	python -W ignore -u utils.py albert-xxlarge-v2 3

clean:
	rm -rf *.log