.PHONY: run test clean

run:
	./probe.sh < params

glue:
	./GLUE/glue.sh < params

test:
	echo "Testing unpruned pretrained LMs"
	python -W ignore -u utils.py

clean:
	rm -f *.log