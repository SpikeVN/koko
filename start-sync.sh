#!/usr/bin/bash

rsync -avzP . triph@xavier:/mnt/sdcard/PhuongBase/koko --exclude='.venv/'
