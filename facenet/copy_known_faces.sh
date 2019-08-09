#!/usr/bin/env bash

dst=src/align/tmp/known_faces
mkdir -p $dst

for i in `ls ../data/known_faces/`; do
  echo $i; 
  name=`echo $i | cut -f 1 -d '.'`; echo $name;
  mkdir -p $dst/$name;
  cp ../data/known_faces/$i $dst/$name/;
done
