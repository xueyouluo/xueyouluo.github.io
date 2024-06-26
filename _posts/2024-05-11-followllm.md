---
layout: post
title: "FollowLLM - 一个关注大模型信息的网站"
date: 2024-02-16
excerpt: "大模型信息"
tags: [LLM, followllm]
comments: true
nlp: false 
---

# 一个跟踪大模型信息的好网站-FollowLLM

## 简介

点击直达 -> https://www.followllm.online

这个网站功能很简单，汇集了每天的arxiv论文、github trending项目、科技公众号文章以及AI产品公众号的文章，方便感兴趣的同学筛选有价值的信息。

由于网页内容是每日实时更新，你还可以结合kimi或者大模型插件，将每日的信息帮你再进行汇总筛选一下。

![image-20240511212507230]({{ site.url }}/assets/img/followllm/image-20240511212507230.png)

![image-20240511214334494]({{ site.url }}/assets/img/followllm/image-20240511214334494.png)

## 功能

### 论文

提供最近5天的论文（只筛选了与LLM相关的），提供简单的标题、摘要翻译和关键词。便于大家发现每天新发的论文有没有感兴趣的。

![image-20240511213332655]({{ site.url }}/assets/img/followllm/image-20240511213332655.png)

每篇点击右上角ℹ️按钮，可以查看来自papers.cool的更详细的总结内容（白嫖的😂），方便进一步筛选。

![image-20240511213508298]({{ site.url }}/assets/img/followllm/image-20240511213508298.png)

### Github

github只收集了每日的趋势，提供分类、关键词和总结，也是便于发现感兴趣的github仓库。



![image-20240511213739637]({{ site.url }}/assets/img/followllm/image-20240511213739637.png)



### 公众号

LLM新闻和AI产品都是公众号内容，暂时只关注了部分公众号，有需求可以随时添加新的分类和公众号。有兴趣可以联系作者。

微信里面公众号现在的推送已经不是按照时间来了，所以，这里也是提供一个简单的汇总入口。

![image-20240511213803038]({{ site.url }}/assets/img/followllm/image-20240511213803038.png)

### 收藏

如果对某篇文章感兴趣，你还可以点击五角星进行收藏。

![image-20240511214455782]({{ site.url }}/assets/img/followllm/image-20240511214455782.png)

### 分享

点击分享按钮，可以复制内容用于分享给好友。

## 其他

这是把原来的Notion数据库变成了网站形式，整体来说算是走了一遍网站搭建的完整流程，不得不吐槽一下国内的开发者真的挺难的：

- 国内域名网站必须ICP备案和公安备案，而国外godaddy买了域名可以直接用。
- 要接个微信登录啥的，还得必须是企业才行，更别说什么付费机制了。

另外前端还是蛮费体力的，由于原来只会简单的flask建个站，作者不得不跟着B站视频学了一遍Vue的前端开发😂。网上虽然有挺多把设计稿或者图片变成前端代码的AI产品，但是生成的代码没点基础改起来还更痛苦。

好啦，敬请期待下一个产品吧。