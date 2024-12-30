#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/30 12:55
# @Author  : chensy
# @Email  : chensiyi010904@163.com
# @File    : main.py
# @Software: PyCharm
from FastLLM import FastLLM

fast_llm = FastLLM()
fast_llm.set_task(["say hi to me！" for i in range(100)])
fast_llm.local_llm_generate()