import csv

def run(name, fields):
    with open('../datasets/' + name + '.csv',  'r') as infile,  open('../datasets/' + name + '-upd.csv',  'w') as outfile:
        # output dict needs a list for new column ordering
        fieldnames = fields
        writer = csv.DictWriter(outfile,  fieldnames=fieldnames)
        # reorder the header first
        writer.writeheader()
        for row in csv.DictReader(infile):
            # writes the reordered rows to the new file
            writer.writerow(row)

def del_last(name):
    with open('../datasets/' + name + '-upd.csv','r') as fin:
        with open('../datasets/' + name + '-upd-r.csv','w') as fout:
            writer=csv.writer(fout)
            for row in csv.reader(fin):
                writer.writerow(row[:-1])

spectrometer_name = 'spectrometer'
spectrometer_fields = ['ID-type', 'Right-Ascension', 'Declination', 'Scale_Factor', 'Blue_base_1', 'Blue_base_2', 'Red_base_1', 'Red_base_2', 'blue-band-flux_1', 'blue-band-flux_2', 'blue-band-flux_3', 'blue-band-flux_4', 'blue-band-flux_5', 'blue-band-flux_6', 'blue-band-flux_7', 'blue-band-flux_8', 'blue-band-flux_9', 'blue-band-flux_10', 'blue-band-flux_11', 'blue-band-flux_12', 'blue-band-flux_13', 'blue-band-flux_14', 'blue-band-flux_15', 'blue-band-flux_16', 'blue-band-flux_17', 'blue-band-flux_18', 'blue-band-flux_19', 'blue-band-flux_20', 'blue-band-flux_21', 'blue-band-flux_22', 'blue-band-flux_23', 'blue-band-flux_24', 'blue-band-flux_25', 'blue-band-flux_26', 'blue-band-flux_27', 'blue-band-flux_28', 'blue-band-flux_29', 'blue-band-flux_30', 'blue-band-flux_31', 'blue-band-flux_32', 'blue-band-flux_33', 'blue-band-flux_34', 'blue-band-flux_35', 'blue-band-flux_36', 'blue-band-flux_37', 'blue-band-flux_38', 'blue-band-flux_39', 'blue-band-flux_40', 'blue-band-flux_41', 'blue-band-flux_42', 'blue-band-flux_43', 'blue-band-flux_44', 'red-band-flux_1', 'red-band-flux_2', 'red-band-flux_3', 'red-band-flux_4', 'red-band-flux_5', 'red-band-flux_6', 'red-band-flux_7', 'red-band-flux_8', 'red-band-flux_9', 'red-band-flux_10', 'red-band-flux_11', 'red-band-flux_12', 'red-band-flux_13', 'red-band-flux_14', 'red-band-flux_15', 'red-band-flux_16', 'red-band-flux_17', 'red-band-flux_18', 'red-band-flux_19', 'red-band-flux_20', 'red-band-flux_21', 'red-band-flux_22', 'red-band-flux_23', 'red-band-flux_24', 'red-band-flux_25', 'red-band-flux_26', 'red-band-flux_27', 'red-band-flux_28', 'red-band-flux_29', 'red-band-flux_30', 'red-band-flux_31', 'red-band-flux_32', 'red-band-flux_33', 'red-band-flux_34', 'red-band-flux_35', 'red-band-flux_36', 'red-band-flux_37', 'red-band-flux_38', 'red-band-flux_39', 'red-band-flux_40', 'red-band-flux_41', 'red-band-flux_42', 'red-band-flux_43', 'red-band-flux_44', 'red-band-flux_45', 'red-band-flux_46', 'red-band-flux_47', 'red-band-flux_48', 'red-band-flux_49', 'LRS-name', 'LRS-class']
run(spectrometer_name, spectrometer_fields)
# del_last(spectrometer_name)

JapaneseVowels_name = 'JapaneseVowels'
JapaneseVowels_fields = ['utterance','frame','coefficient1','coefficient2','coefficient3','coefficient4','coefficient5','coefficient6','coefficient7','coefficient8','coefficient9','coefficient10','coefficient11','coefficient12','speaker']
run(JapaneseVowels_name, JapaneseVowels_fields)


helena_name = 'helena'
helena_fields = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','class']
run(helena_name, helena_fields)
