#!/usr/bin/env python
import sys

fn=sys.argv[1]

match = 0
noMatch = 0

for line in sys.stdin:
	fields = line.split()
	#print "FIELD VALUE is {}".format(fields[2])
	if fields[2] is '1':
		#print "MATCH"
		match = match+1
	else:
		#print "NO MATCH"
		noMatch = noMatch+1

print '{3}, {4}, Matches {0}, No matches {1}, Total {2}, '.format(match,noMatch,match+noMatch,float(match/float(match+noMatch)),fn)

    
	

