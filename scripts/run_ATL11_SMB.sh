#!/bin/bash

dest_dir=/Volumes/ice2/ben/MAR/ATL11_with_corrections/rel002
#xover_file=$dest_dir/test_xovers.h5


xover_file=$dest_dir/rel002_crossover_data.h5
AT_file=$dest_dir/rel002_dump_every_2nd.h5

echo "Crossover_file: "$xover_file
echo "Along-track file: "$AT_file

[ -f $xover_file ] && [ -f $AT_file ] && echo "yep"
#exit
=======
# dest_dir=/Volumes/ice2/ben/MAR/ATL11_with_corrections/temp
dest_dir=/Volumes/ice3/tyler/MAR/ATL11_with_corrections/rel002a

atl11_files=( "rel002_crossover_data.h5" "rel002_dump_every_2nd.h5" )

for i in "${atl11_files[@]}"
do

    # check if running crossover or along track file
    if [[ $i =~ "crossover" ]]
    then
        j="Crossover"
    else
        j="Along-track"
    fi

    # test if file exists
    atl11_file="$dest_dir/$i"
    if [ -f "$atl11_file" ]
    then
        printf "ATL11 %s File: %s\n" $j $atl11_file
    else
        printf "ATL11 %s File %s not found!\n" $j $atl11_file
        continue
    fi

    echo "SMB"
    append_SMB_ATL11.py --directory /Volumes/ice3/tyler/ \
        --model MERRA2-hybrid --region GL $atl11_file

    # # echo "SMB averages"
    # # append_SMB_averages_ATL11.py --directory /Volumes/ice3/tyler/ \
    # #     --model MAR --region GL --year 2000 2019 $atl11_file

    echo "SMB mean"
    append_SMB_mean_ATL11.py --directory /Volumes/ice3/tyler/ \
       --model MAR --region GL --year 1980 1995 $atl11_file

    # # append_SMB_mean_ATL11.py --directory /Volumes/ice3/tyler/ \
    # #     --model MAR --region GL --year 2000 2019 $atl11_file

done

echo "SMB"
append_SMB_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR RACMO MERRA2-hybrid --region GL $AT_file
append_SMB_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR RACMO MERRA2-hybrid --region GL $xover_file

#echo "SMB averages"
#append_SMB_averages_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 2000,2019 $AT_file
#append_SMB_averages_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 2000,2019 $xover_file

echo "SMB mean"
append_SMB_mean_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 1980,1995 $AT_file
append_SMB_mean_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 1980,1995 $xover_file

#append_SMB_mean_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 2000,2019 $AT_file
#append_SMB_mean_ATL11.py --directory /Volumes/ice1/tyler/ --model MAR --region GL --year 2000,2019 $xover_file
