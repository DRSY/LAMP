run:
	./probe.sh < params

test:
	echo "Testing unpruned pretrained LMs"
	python -W ignore -u utils.py

clean:
	rm -f *.log