# This file is part of UDPipe <http://github.com/ufal/udpipe/>.
#
# Copyright 2016 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys

from ufal.udpipe import Model, Pipeline, ProcessingError # pylint: disable=no-name-in-module
'''

if len(sys.argv) < 4:
    sys.stderr.write('Usage: %s input_format(tokenize|conllu|horizontal|vertical) output_format(conllu) model_file\n' % sys.argv[0])
    sys.exit(1)
'''
modelPath = '../resources/Model/italian-isdt-ud-2.5-191206.udpipe'  # was sys.argv[3]
sys.stderr.write('Loading model: ')
model = Model.load(modelPath)

if not model:
    sys.stderr.write("Cannot load model from file '%s'\n" % modelPath)
    sys.exit(1)
sys.stderr.write('done\n')


# sys.argv[1] = input file, sys.argv[2] = output file
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
error = ProcessingError()

# Read whole input
text = 'Questo è un testo di prova per testare UDPipe'

# Process data
processed = pipeline.process(text, error)
if error.occurred():
    sys.stderr.write("An error occurred when running run_udpipe: ")
    sys.stderr.write(error.message)
    sys.stderr.write("\n")
    sys.exit(1)
sys.stdout.write(processed)