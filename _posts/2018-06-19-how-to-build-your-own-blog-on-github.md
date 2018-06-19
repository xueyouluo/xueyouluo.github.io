---
layout: post
title: "Build your own blog on github"
date: 2018-06-19
excerpt: "how to build a blog on gitbug pages."
tags: [blog,github pages,moon,Jekyll]
comments: true
---

I found there are lots of fantastic and beautiful blogs based github pages. I also want to build my own blog website, but I knew little about html and css. 

Thanks to [Moon](https://github.com/TaylanTatli/Moon), it makes things much easier. You don't need to know much about css, html and Jekyll to create a beautiful blog, the author has done all the work for you, you can focus on writing blogs.

I wrote down some basic steps to setup a blog, in case it maybe helpful for others.

## Create a repository

You need a github account firstly.

Then, head over to GitHub and create a new repository named username.github.io, where username is your username (or organization name) on GitHub. 

For example, my github account is xueyouluo, so my new repository is xueyouluo.github.io.

You can refer to [github pages](https://pages.github.com/) to find more details.

## Jekyll

What is Jekyll? In short words, it is a static site generator. You can create your blogs as markdown files.

Install Jekyll on your local machine to debug locally. You can find the installation tutorials on [Jekyll website](https://jekyllrb.com/docs/installation/#ubuntu).

You can find more details about what a Jekyll website project looks like by google. The most important things are:
* _config.yml
    * important config file
* _posts
    * all your blogs are listed here
    * they are named by YYYY-MM-DD-Blog-Title.md
* _layouts
    * Html files used to render your blogs or your homepage
* assets
    * images can be placed here
    * css and js files, but we don't care

## Copy & Paste

Clone the [Moon](https://github.com/TaylanTatli/Moon) project, and copy all the files to your repository, for example xueyouluo.github.io.

Modify the _config.yml file, such as change the title, disqus\_shortname. you can find more details from the [tutorial](https://taylantatli.github.io/Moon/moon-theme/). 

Remember to run `bundle update`.

Remove all the files under the folder _posts, and create a new md file and write something. 

You should change the url in _config.yml to http://localhost:4000 if you want to debug locally.

> If your want to enable comments, go to https://disqus.com/home/ to register a new disqus\_shortname for yourself, and change the disqus\_shortname in the _config.yml to yours.

## Start the server

Run following command to start the serve: 
```bash
bundle exec jekyll serve --host=0.0.0.0
```

Then you can browser your blog on your local machine.


## Save & Commit

Commit all the changes to your repository. That all!