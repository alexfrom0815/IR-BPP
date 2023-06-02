rm *.user
git add --all
git commit -m "$1"
git push -u origin $2
