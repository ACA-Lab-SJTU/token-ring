#!/bin/bash
rm ./output/*

torchrun --nproc_per_node=2 test/test_zigzag_token_ring_func.py &
wait
