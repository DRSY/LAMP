.PHONY: run test clean glue

run:
	./probe.sh < params

glue:
	./GLUE/glue.sh < ./GLUE/params

test:
	echo "Testing unpruned pretrained LMs"
	python -W ignore -u utils.py

clean:
	rm -f *.log