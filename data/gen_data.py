#! /usr/bin/python3

from xeger import Xeger

gen = Xeger(limit=26)
ch = gen.xeger(r'[a-z]')
print(ch)