#! /bin/bash -eux
set -o pipefail

# Test workflow from fast5 files to trained model using flip-flop remapping with refs extracted from sam
# This is done with just a few reads so the model
# won't be useful for anything.
# This script must be executed with the current directory being the taiyaki base directory

echo ""
echo "Test of extract-ref-from-sam followed by flip-flop remap and basecall network training starting"
echo ""

# Execute the whole workflow, extracting references, generating per-read-params and mapped-read files and then training
READ_DIR=test/data/reads
SAM_DIR=test/data/aligner_output
# The |xargs puts spaces rather than newlines between the filenames
SAMFILES=$(ls ${SAM_DIR}/*.sam |xargs)
REFERENCEFILE=test/data/genomic_reference.fasta

echo "SAMFILES=${SAMFILES}"
echo "REFERENCEFILE=${REFERENCEFILE}"

TAIYAKI_DIR=`pwd`
RESULT_DIR=${TAIYAKI_DIR}/RESULTS/train_remap_samref
envDir=${envDir:-$TAIYAKI_DIR}

rm -rf $RESULT_DIR
rm -rf ${TAIYAKI_DIR}/RESULTS/training_ingredients

#TAIYAKIACTIVATE=(nothing) makes the test run without activating the venv at each step. Necessary for running on the git server.
make -f workflow/Makefile READDIR=${READ_DIR} TAIYAKI_ROOT=${TAIYAKI_DIR} BAMFILE="${SAMFILES}" REFERENCEFILE=${REFERENCEFILE} SEED=1 TAIYAKIACTIVATE= train_remap_samref envDir=${envDir}

# Check that training log exists and has enough rows for us to be sure something useful has happened


traininglog_lines=`wc -l ${RESULT_DIR}/model.log | cut -f1 -d' '`
echo "Number of lines in training log: ${traininglog_lines}"
if [ "$traininglog_lines" -lt "9" ]
then
    echo "Training log too short- training not started properly"
    exit 1
fi

echo ""
echo "Test of extract-ref-from-sam followed by flip-flop remap and basecall network training completed successfully"
echo ""
