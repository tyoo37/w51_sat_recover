#!/bin/bash
mem=16gb
taskname=making_training_set


#daoloop=("--daophot --skip-crowdsource" " ")
#mem=32gb
#for filter in F140M; do
#
#for filter in F1280W F2100W; do
#for filter in F560W F770W F1000W F1280W F2100W; do
#for filter in F140M F162M F182M F187N F210M F335M F360M F410M F405N F480M ; do
#    sbatch --job-name=add_sat_star_find-${filter} --output=add_sat_star_find-${filter}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/from_red/w51/w51_catalog/py/add_saturated_stars.py --filter=${filter}"
#done
for filter in F140M F162M F182M F187N F210M F335M F360M F410M F405N F480M ; do
    sbatch --job-name=making_training_set-${filter} --output=making_training_set-${filter}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_sat_recover/py/making_training_set.py --filter=${filter} --target=w51"
done
for filter in F560W F770W F1000W F1280W F2100W; do
#for filter in F1280W F2100W; do
#for filter in F560W F770W F1000W F1280W F2100W; do
    sbatch --job-name=making_training_set-${filter} --output=making_training_set-${filter}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_sat_recover/py/making_training_set.py --filter=${filter} --target=w51_miri"
done


