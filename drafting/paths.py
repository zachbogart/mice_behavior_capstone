import os

home = os.path.expanduser("~")
apps = "/usr/applications"
mnt = '/mnt'

paths = {
    #folders
    "home" : home,
    "drosophila_data" : os.path.join(home, "science/data/drosophila_data/"),
    "locus_scaffolds" : os.path.join(home, "science/backups/sapo/PD_seqproj/fosmids/locus_scaffolds/"),
    "temp" : os.path.join(home, "tmp"),
    "py_testdata" : os.path.join(home, "science/py_util/unit_test_data"),
    "trees" : os.path.join(home, "science/data/trees"),
    "LAGAN_DIR" : "/home/brant/lagan20",
    'tba_dir' : os.path.join(apps, 'multiz'),
    'UCSC_drosophila_align': os.path.join(home, 'science/data/drosophila_data/UCSC/alignments/clean'),
    'UCSC_drosophila_sequence': os.path.join(home, 'science/data/drosophila_data/UCSC/sequence'),

    #apps
#    "mlagan" : "/eisenlab/software/local/lagan20/mlagan",
    "mlagan" : "/home/brant/lagan20/mlagan",
    'TBA' : os.path.join(apps, 'multiz', 'tba'),
    'roast' : os.path.join(apps, 'multiz', 'roast'),
    'blat' : os.path.join(apps,'bin','blat'),
    'bl2seq' : 'bl2seq',

    #files
    "PD_tree" : os.path.join(mnt, "trees", "PD_all_sp+dros.newick")
    }
