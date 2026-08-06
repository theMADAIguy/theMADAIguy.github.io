---
layout: page
title: Blog
permalink: /blog/
order: 4
---

{% if site.posts.size > 0 %}
<ul class="pub-list">
{% for post in site.posts %}
<li>
<span class="pub-title"><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></span>
<span class="pub-meta">{{ post.date | date: "%B %-d, %Y" }}</span>
</li>
{% endfor %}
</ul>
{% else %}
<p><em>No posts yet &mdash; check back soon.</em></p>
{% endif %}
