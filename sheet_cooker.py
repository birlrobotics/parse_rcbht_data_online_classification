folder_name_pattern = '%s classification of %s-arm %s data'

classification_type = ["state", "SF"]
arm_type = ["one", "two"]
data_type = ["SIM", "REAL"]

for ct in classification_type:
    for at in arm_type:
        for dt in data_type:
            now_folder = folder_name_pattern%(ct, at, dt)
            print now_folder
            import os
            try:
                files = os.listdir(now_folder)
            except OSError as e:
                print "not found"
                continue
           
            rows = []
            for f in files:
                if f.endswith(".txt"):
                    print f
                else:
                    continue

                items = f[:-4].split('_')
                kernel = items[1] 
                C = eval(items[3])
                
                lines = open(os.path.join(now_folder, f)).readlines()
                d = dict([l.split(":") for l in lines])

                last_mean = eval(d['y_for_mean'])[-1]
                last_max = eval(d['y_for_max'])[-1]
                last_min = eval(d['y_for_min'])[-1]

                rows.append({"kernel":kernel, "C":C, "min":last_min, "mean":last_mean, "max":last_max})

            print rows
            rows = sorted(rows, key=lambda x:(x["kernel"], x["C"]))

            csv_file = open("./tmp_dir2/"+now_folder+".txt", "w")

            import csv
            fieldnames = ['kernel', 'C', 'min', 'mean', 'max']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

            


