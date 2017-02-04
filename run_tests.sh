#!/usr/bin/env bash
export PYTHONPATH=${PWD}

for SCRIPT in ./benchmark/*
	do
		if [ ${SCRIPT: -3} == ".sh" ]
		then
			${SCRIPT}
		fi
	done