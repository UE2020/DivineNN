#!/bin/bash
git checkout --orphan temp $1 # create a new branch without parent history
git commit -m "Truncated history" # create a first commit on this branch
git rebase --onto temp $1 main # now rebase the part of master branch that we want to keep onto this branch
git branch -D temp # delete the temp branch

# The following 2 commands are optional - they keep your git repo in good shape.
git prune --progress # delete all the objects w/o references
git gc --aggressive # aggressively collect garbage; may take a lot of time on large repos