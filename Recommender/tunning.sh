for weight in 1.0 2.0
do
	for dim in 2000 1000
	do
		for reg_u in 0.00 0.01 0.03 0.05 0.1 0.15 0.2 0.3 0.5
		do
			for reg_l in 0.00 0.01 0.03 0.05 0.1 0.15 0.2 0.3 0.5
			do
				for keepProb in 0.2 0.4
				do
					name=FQ_parameters
					python AE.py 16 0.0005 $dim $weight $reg_u $reg_l $keepProb 'Foursquare'>> $name.log
				done
			done
		done
	done
done
