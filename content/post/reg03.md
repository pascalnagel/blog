---
title: "Feature reconstruction in the Porto Seguro Kaggle competition"
date: 2017-12-07T20:54:06+01:00
draft: false
coverImage: images/seguro2.jpg
coverSize: partial
thumbnailImage: images/seguro.jpg
thumbnailImagePosition: left
---

The [Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) Kaggle competition featured data, which was anonymized to the point, where the actual meaning of the features was obscured. So well in fact, that during the competition only very few features were explained, making manual feature engineering essentially impossible. As a result, the winning solutions were based on neural networks, which learned good representations and avoided feature engineering entirely.

I managed to reverse engineer one the most important features of the data set and posted the following [findings](https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03) in a Kaggle hosted kernel.

## The Reconstruction

<html>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The closed form guesser of <a href="https://www.wolframalpha.com/">Wolfram|Alpha</a>, applied to some high precision values of ps_reg_03 reveals the following pattern:
$$
\text{ps_reg_03}=\frac{\sqrt{I}}{40},\quad \text{ with } I\in \mathbb{N^+}.
$$</p>
<p>In other words:
$$
I=\left(40*\text{ps_reg_03}\right)^2
$$</p>
<p>yields an integer. ps_reg_03 is therefore likely a categorical feature or a combination of combinatorical features.</p>
<p>Lets confirm this pattern for the full data set:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Load training data</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;../input/train.csv&#39;</span><span class="p">)</span>
<span class="c1"># and remove nan values for now</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="p">(</span><span class="mi">40</span><span class="o">*</span><span class="n">x</span><span class="p">)</span><span class="o">**</span><span class="mi">2</span><span class="p">)</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0      825.0
1      939.0
3      540.0
4     1131.0
5     8706.0
6      610.0
7      590.0
8     1300.0
9     8587.0
10    1013.0
Name: ps_reg_03_int, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This looks promising and indeed all ps_reg_03_int are very close to an integer.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Actually convert ps_reg_03_int to integer</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">)</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As a cross-check, let us count the number of unique values of ps_reg_03 and the number of integer categories:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Unique values of ps_reg_03: &quot;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of integer categories: &quot;</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Unique values of ps_reg_03:  5012
Number of integer categories:  5012
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now let's have a look at their distribution:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="p">):</span> <span class="n">ax</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s1">&#39;log&#39;</span><span class="p">)</span> <span class="c1">#Set y-axis to log-scale</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">)</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03_int&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Integer&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.text.Text at 0x7f0f537856d8&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that these integers are probably not the <a href="https://en.wikipedia.org/wiki/Municipalities_of_Brazil">municipalities of Brazil</a> (1-5570), since values as large as 25000 exist. The skewness of the distribution supports this, although the data could have been sorted and municipalities relabeled.</p>
<p>It seems more likely that something like <a href="https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41200">Glimmung's interpretation</a> could be correct:</p>
<p>$$
\text{ps_reg_03}=\frac{\sqrt{27M + F}}{40},
$$</p>
<p>with F being the <a href="https://en.wikipedia.org/wiki/Subdivisions_of_Brazil">federative unit of Brazil</a> (1-27) and M being the municipal number inside that administrative region, which can grow as large as a few hundred. Since there are only 27 federative units, the above equation yields a unique value for each combination of M and F.</p>
<p>For instance, the municipality 76 in unit 8 would yield the integer and ps_reg_03:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Integer: &quot;</span><span class="p">,</span> <span class="mi">27</span><span class="o">*</span><span class="mi">76</span><span class="o">+</span><span class="mi">8</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;ps_reg_03: &quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">sqrt</span><span class="p">(</span><span class="mi">27</span><span class="o">*</span><span class="mi">76</span><span class="o">+</span><span class="mi">8</span><span class="p">)</span><span class="o">/</span><span class="mi">40</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Integer:  2060
ps_reg_03:  1.13468057179
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This would explain the skewness, since lower municipalities are much more common (every federative unit has a municipality #1, but not #270).</p>
<p>Assuming for now, that the above equation is the correct interpretation, let's reconstruct M and F of each integer category:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">recon</span><span class="p">(</span><span class="n">reg</span><span class="p">):</span>
    <span class="n">integer</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">((</span><span class="mi">40</span><span class="o">*</span><span class="n">reg</span><span class="p">)</span><span class="o">**</span><span class="mi">2</span><span class="p">))</span> <span class="c1"># gives 2060 for our example</span>
    <span class="k">for</span> <span class="n">f</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">28</span><span class="p">):</span>
        <span class="k">if</span> <span class="p">(</span><span class="n">integer</span> <span class="o">-</span> <span class="n">f</span><span class="p">)</span> <span class="o">%</span> <span class="mi">27</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="n">F</span> <span class="o">=</span> <span class="n">f</span>
    <span class="n">M</span> <span class="o">=</span> <span class="p">(</span><span class="n">integer</span> <span class="o">-</span> <span class="n">F</span><span class="p">)</span><span class="o">//</span><span class="mi">27</span>
    <span class="k">return</span> <span class="n">F</span><span class="p">,</span> <span class="n">M</span>

<span class="c1"># Using the above example to test</span>
<span class="n">ps_reg_03_example</span> <span class="o">=</span> <span class="mf">1.13468057179</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Federative Unit (F): &quot;</span><span class="p">,</span> <span class="n">recon</span><span class="p">(</span><span class="n">ps_reg_03_example</span><span class="p">)[</span><span class="mi">0</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Municipality (M): &quot;</span><span class="p">,</span> <span class="n">recon</span><span class="p">(</span><span class="n">ps_reg_03_example</span><span class="p">)[</span><span class="mi">1</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Federative Unit (F):  8
Municipality (M):  76
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's now apply this to the data set. The function above, despite being horribly inefficient, runs within a few seconds.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_F&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">recon</span><span class="p">(</span><span class="n">x</span><span class="p">)[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_M&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">recon</span><span class="p">(</span><span class="n">x</span><span class="p">)[</span><span class="mi">1</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">&#39;ps_reg_03&#39;</span><span class="p">,</span> <span class="s1">&#39;ps_reg_F&#39;</span><span class="p">,</span> <span class="s1">&#39;ps_reg_M&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>    ps_reg_03  ps_reg_F  ps_reg_M
0    0.718070        15        30
1    0.766078        21        34
3    0.580948        27        19
4    0.840759        24        41
5    2.332649        12       322
6    0.617454        16        22
7    0.607248        23        21
8    0.901388         4        48
9    2.316652         1       318
10   0.795692        14        37
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The administrative regions and municipalities are distributed as follows:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_yscale</span><span class="p">(</span><span class="s1">&#39;log&#39;</span><span class="p">)</span> <span class="c1">#Set y-axis to log-scale only for M</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_M&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Municipality (M)&#39;</span><span class="p">)</span>
<span class="n">train</span><span class="p">[</span><span class="s1">&#39;ps_reg_F&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">ax</span><span class="o">=</span><span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="n">bins</span><span class="o">=</span><span class="mi">27</span><span class="p">)</span>
<span class="n">ax</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s1">&#39;Federative Unit (F)&#39;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.text.Text at 0x7f0f536177f0&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The number of policy holders seems to be somewhat uniformly distributed over the federative units. The municipality is skewed right as expected.</p>

</div>
</div>
</div>

</html>


