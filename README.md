# Learngin-Github
$ git clone git@github.com:kos00pas/Learngin-Github.git
			--->Cloning into 'Learngin-Github'...

$ git status
			--->fatal: not a git repository (or any of the parent directories): .git


 $ ls
			--->Learngin-Github

$ cd Learngin-Github/
			


$ ls
			--->README.md



 $ git status
			--->On branch main
				Your branch is up to date with 'origin/main'.

				Changes not staged for commit:
				  (use "git add/rm <file>..." to update what will be committed)
				  (use "git restore <file>..." to discard changes in working directory)
					deleted:    The_Mfcc_v3.py
					deleted:    index.html

				no changes added to commit (use "git add" and/or "git commit -a")

 $ git add .


$ git commit -m "delete"
			[main d939e04] delete
			 2 files changed, 232 deletions(-)
			 delete mode 100644 The_Mfcc_v3.py
			 delete mode 100644 index.html

$ git push
			Enumerating objects: 3, done.
			Counting objects: 100% (3/3), done.
			Delta compression using up to 4 threads
			Compressing objects: 100% (1/1), done.
			Writing objects: 100% (2/2), 223 bytes | 223.00 KiB/s, done.
			Total 2 (delta 0), reused 0 (delta 0), pack-reused 0
			To github.com:kos00pas/Learngin-Github.git
			   31c7ab0..d939e04  main -> main
