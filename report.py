"""Build the report only from completed course experiments and frozen test results."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

BASE=Path(__file__).resolve().parent


def load(p):return json.loads(Path(p).read_text())
def pct(x):return f'{100*x:.2f}%'
def summary(d):return pct(d['mean'])+(f" +/- {100*d['sd']:.2f} pp" if d['sd'] is not None else '')


def build(repo_url,font_dir):
    if not repo_url.startswith('https://github.com/Dorrie-ge/'):
        raise ValueError('Supply the verified public HW3 repository URL')
    course=BASE/'results/course';r=load(BASE/'results/report_metrics.json');s=load(course/'selection.json')
    tests=load(course/'heldout_test.json');checks=load(BASE/'results/validation/algorithm_validation.json')
    diagnostics=load(BASE/'results/nonstandard_diagnostics.json')['runs']
    adapters=load(BASE/'results/validation/adapter_validation.json')
    crosschecks=load(BASE/'results/validation/course_crosschecks.json')
    edges=load(BASE/'results/validation/edge_validation.json')
    assert edges['all_passed'] and len(edges['checks'])==10
    assert adapters['json_hdf5_equivalence'] and adapters['json_npy_equivalence']
    assert all(row['max_history_metric_error']==0 for row in crosschecks['checks'])
    for name in s['test_runs']:
        if not (course/name/'COMPLETE').exists():raise ValueError('Incomplete run '+name)
    if len(r['baseline'])!=4 or len(r['noise'])!=5:raise ValueError('Missing comparison')
    fonts=Path(font_dir)
    for name,file in [('TNR','Times New Roman.ttf'),('TNR-Bold','Times New Roman Bold.ttf'),('TNR-Italic','Times New Roman Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-Bold')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText']:styles[name].fontName='TNR';styles[name].fontSize=10.5;styles[name].leading=14
    styles.add(ParagraphStyle(name='TitleX',fontName='TNR-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#142b49'),spaceAfter=14))
    styles.add(ParagraphStyle(name='H',fontName='TNR-Bold',fontSize=16,leading=20,textColor=colors.HexColor('#142b49'),spaceAfter=10))
    styles.add(ParagraphStyle(name='Sub',fontName='TNR-Bold',fontSize=11.5,leading=15,spaceBefore=8,spaceAfter=5))
    styles.add(ParagraphStyle(name='SmallX',fontName='TNR',fontSize=9,leading=11.5,spaceAfter=7))
    story=[];figs=course/'figures'
    def p(text,style='BodyText'):
        story.append(Paragraph(text,styles[style]));story.append(Spacer(1,6))
    def h(text):p(text,'H')
    def page():
        while story and isinstance(story[-1],Spacer):
            story.pop()
        story.append(PageBreak())
    def table(rows,widths=None):
        rows=[[Paragraph(escape(str(x)),styles['SmallX']) for x in row] for row in rows]
        t=Table(rows,colWidths=widths or [492/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9cabbc')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce3eb')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,10))
    def img(name,width=492,height=None):
        im=Image(str(figs/name));ratio=im.imageHeight/im.imageWidth
        if height is not None and width*ratio>height:width=height/ratio
        im.drawWidth=width;im.drawHeight=width*ratio;story.append(im);story.append(Spacer(1,5))
    chosen=s['chosen_baseline'];best=next(row for row in r['baseline'] if row['fraction']==chosen['fraction'] and row['epochs']==chosen['epochs'])
    b=s['chosen_noise_scale'];noise=next((row for row in r['noise'] if row['scale']==b),None)
    representative=chosen['runs'][0];ray_name=s['ray_runs'][0]
    p('Assignment 3<br/>Federated Learning and Differential Privacy','TitleX')
    p('GitHub username: <b>Dorrie-ge</b><br/>Repository: <b>'+escape(repo_url.rsplit('/',1)[-1])+'</b><br/>Public source: <link href="'+repo_url+'">'+repo_url+'</link>')
    p('My objective and results','Sub')
    p('I implement FedAvg directly in PyTorch and use Ray Core actors for parallel local updates. I train on the exact 100-writer Federated EMNIST archive downloaded from this course\'s Canvas Files, preserve its held-out test partition, and evaluate a 784-128-62 fully connected classifier. I base the numbers below on saved training histories, checkpoints, validation controls and frozen test evaluations.')
    table([['Experiment','Held-out accuracy','Evidence'],['Chosen serial FedAvg',summary(best['test_accuracy']),'Three seeds, 120 rounds each'],['Ray, four clients per round',pct(r['ray']['test_accuracy']),'One seed, 120 rounds'],['Chosen nonzero noise scale',('b='+str(b)+': '+summary(noise['test_accuracy'])) if noise else 'No nonzero scale met my utility rule','Three seeds; clean held-out images']],[155,175,162])
    p('The +/- values are sample standard deviations across three training seeds, in percentage points; they are not confidence intervals. Ray uses one seed, so its result has no across-training-seed uncertainty estimate. I do not treat these modest-budget runs as optimal or fully converged models.')
    p('How I organize the evidence','Sub')
    p('I distinguish exact algorithm identities, controlled learning experiments, and empirical generalization. My validation includes a pooled-gradient identity, unequal-client weighting, zero-update controls, Laplace moments, label permutation on real writers, local-epoch stress, and matched serial/Ray trajectories. I retain every round and the first ten rounds of client-update records in JSON/JSONL, with reproduction commands in the final pages and README.')
    p('Execution: November 14, 2026. Training uses CPU SGD; each Ray actor reserves one CPU and zero GPUs.','SmallX')
    page();h('1. My data, partitions and writer heterogeneity')
    d=r['data'];p(f"The provided training array contains {d['train_plus_val']:,} images across 100 users; the separate test array contains 3,621 images. Each image has 28 by 28 float32 pixels already in [0,1]. I flatten the images without changing orientation or polarity, keep integer labels 0-61, and retain all 62 logits even when a client lacks a class.")
    p('I split each client independently using seed 2026 and a stable writer-specific seed: floor(0.8 n) training examples and the remainder validation. This produces <b>25,418 training and 6,407 validation images</b>; train/validation indices are disjoint and collectively exhaustive. No client shares its images with another client; the server aggregates model states and scalar sample-weighted summaries.')
    img('class_distributions.png',height=365)
    p('Figure 1. Class counts for the complete provided local-data collection and clients 000-004. These five writers were selected by fixed index, not by favorable performance. Classes are labeled with the dataset\'s supplied numerical IDs.','SmallX')
    p(f"User sample sizes range from {d['min_client_samples']} to {d['max_client_samples']}; global class counts range from {d['min_class_samples']} to {d['max_class_samples']}. Mean writer-to-global label total-variation distance is {d['mean_client_total_variation']:.3f}, with maximum {d['max_client_total_variation']:.3f}. I therefore expect both sampling noise and non-IID client updates; a global accuracy alone can hide weaker writers.")
    page();h('2. My FedAvg implementation and local evaluation')
    p('My model has two affine layers: 784 inputs to 128 hidden units, ReLU, then 62 output logits. I use cross-entropy with minibatch size 32 and local SGD at learning rate 0.05; local optimizer state is reset each round. With two bias vectors, the model has 108,478 trainable parameters. There is no specialized federated-learning package.')
    p('At round t I sample m=floor(C times 100) clients uniformly without replacement, copy the same global parameters to each selected client, and perform E complete local epochs. For selected client k with n_k training samples, I aggregate <b>w(t+1) = sum_k [n_k / sum_j n_j] w_k</b>. I weight by actual training samples, not by the number of minibatches, number of epochs, or an unweighted average of clients.')
    p('After aggregation I evaluate the current global model locally on every client. Each client returns loss sums, correct counts and sample counts; global loss/accuracy are totals divided by total samples. I retain perturbed-training and clean-training metrics separately for noisy runs. Clean validation never updates the model, and the training function never opens the held-out test directory.')
    img(representative+'_curves.png',height=250)
    p('Figure 2. Training/validation loss and accuracy for the representative seed-17 selected serial model. Round zero evaluates the common initialization. Curves are raw measurements, not smoothed or constrained to be monotone.','SmallX')
    p('Edge cases I handle','Sub')
    p('I reject clients with fewer than two samples, malformed labels, out-of-range clean pixels, invalid C, and nonpositive aggregation weights. Client batch sizes need not divide local dataset sizes: the final short minibatch is retained. Missing local classes are valid; I do not stratify away writer heterogeneity. Random sampling uses a separate generator, so model evaluation cannot change the client schedule.')
    page();h('3. How participation and local epochs change learning')
    img('ce_comparison.png',height=250)
    p('Figure 3. I compare C in {0.04,0.10} and E in {1,3}, never exceeding 10% participation. Lines are means across seeds 17, 29 and 43; bands show sample SD. All runs use identical split policies, 120 rounds, minibatches and learning rates.','SmallX')
    rows=[['C / clients per round','Local E','Final train accuracy','Final validation accuracy']]
    for row in r['baseline']:rows.append([f"{row['fraction']:.2f} / {int(row['fraction']*100)}",row['epochs'],summary(row['train_accuracy']),summary(row['val_accuracy'])])
    table(rows,[115,55,161,161])
    p(f"I select C={chosen['fraction']:.2f}, E={chosen['epochs']} by its mean final validation accuracy, {pct(chosen['mean_validation'])}. I freeze this choice before test evaluation. Increasing E and C improves the selected combination in my budget, but E also increases local computation and C increases the number of transmitted model states. Fixed communication rounds do not equal fixed computation.")
    p('The lower-participation, low-E runs show substantial seed sensitivity. The standard deviation is part of the result, rather than a reason to discard a seed. I report only the chosen baseline on held-out test data; I do not use held-out accuracy to rank the C/E sweep. My limited learning-rate pilot comparisons are retained as pilots and are not included in these three-seed means.')
    p('With 4-byte parameters, one full model is 433,912 bytes; 10 clients require about 8.68 MB of model upload plus download per round, versus about 3.47 MB for 4 clients, excluding protocol overhead. This makes the accuracy/communication tradeoff concrete.','SmallX')
    page();h('4. Ray Core: four selected client actors per round')
    p('I instantiate exactly four Ray actors for the four selected clients at each round. Each actor owns one client partition, creates its local model, performs local SGD, and returns a state and a sample count. I reserve num_cpus=1 and num_gpus=0 for each actor, wait until all four are ready, send the common state through Ray\'s object store, aggregate the returned states, and kill the actors before the next round. I create no actors for the other 96 users.')
    img(ray_name+'_curves.png',height=225)
    p('Figure 4. The required training/validation loss and accuracy trajectories for four-client Ray FedAvg, 120 communication rounds.','SmallX')
    table([['Validation target','Observed'],['Final validation accuracy',pct(r['ray']['final']['val_accuracy'])],['Held-out test accuracy',pct(r['ray']['test_accuracy'])],['Matched serial/Ray parameter difference',f"{r['ray']['max_parameter_error']:.3g} maximum absolute error"],['Matched serial/Ray metric difference',f"{r['ray']['max_metric_error']:.3g} maximum absolute error"],['Peak active update intervals',f"{max(r['ray']['peak_overlaps'])} overlapping client updates"]],[245,247])
    p('The serial comparison uses the same seed, data, E and four selected clients, not the 10-client headline baseline. Matching both model parameters and every round\'s scalar losses/accuracies makes this a stronger orchestration check than observing that both programs finish. The interval overlap confirms parallel execution; it does not establish a speedup, since process startup and memory overhead matter.')
    page();h('5. My exact algorithm and perturbation controls')
    rows=[['Controlled prediction','Observed evidence'],['Unequal sizes: weights 2 and 10 with counts 1 and 3 must average to 8',f"Observed {checks['unequal_client_weighting']['observed']}; unweighted average would be 6"],['One full-batch local SGD step equals pooled SGD on unequal-size synthetic clients',f"Maximum parameter difference {checks['full_batch_gradient_identity']['max_abs_error']:.3g}; tolerance 1e-6"],['E=0 or learning rate=0 cannot change parameters','Both maximum parameter differences are exactly zero'],['Zero Laplace scale preserves inputs','Bitwise equality'],['Laplace(0,0.2): mean 0, variance 0.08',f"100,000 draws: mean {checks['laplace_moments']['mean']:.6f}; variance {checks['laplace_moments']['variance']:.6f}"],['A known separable synthetic task should learn',f"Validation loss {checks['separable_task']['initial']['val_loss']:.4f} -> {checks['separable_task']['final']['val_loss']:.4f}; accuracy 100%"],['Four Ray updates should equal serial updates','Maximum parameter difference 0; four overlapping update intervals']]
    table(rows,[270,222])
    p('Why these tests are informative','Sub')
    p('The unequal-client example detects the specific error of averaging users equally. The full-batch identity isolates FedAvg weighting and the gradient update under a condition where their correct answer is known; I do not expect this identity after multiple local steps. Zero-E and zero-learning-rate counterfactuals detect updates that happen outside the intended optimizer. The separable task tests actual learning through the full selection/aggregation path, not just arithmetic.')
    p('I keep the synthetic fixtures distinct from handwritten-image evaluation. In that fixture I also place an unreadable file in the test directory: the successful training path demonstrates that the directory was not read. The Laplace moment check tests both the scale convention and independence-driven aggregate behavior; b is the Laplace scale, not its standard deviation. All these experiments are reproducible with python validate.py --ray.')
    p('My two auxiliary checks now have standalone scripts. validate_adapters.py reproduces JSON/HDF5/NPY equivalence for 100 synthetic clients, exact feature/label indexing, disjoint and exhaustive 8/2 splits, and same-seed repetition. validate_crosschecks.py verifies three b=0/clean checkpoint pairs and all 121 history points per pair: every maximum parameter and metric difference is 0. These are supporting checks; my real-data counterfactuals below provide the substantive learning validation.','SmallX')
    page();h('6. Real-data interventions: does learning depend on labels?')
    img('course_controls.png',height=250)
    controls=r['controls'];p(f"Figure 5. Left: I shuffle training labels independently within each writer, retaining each writer's class histogram and leaving validation images/labels unchanged. Right: I increase E from 3 to 10 under the same seed, selected-client schedule and 40-round communication budget.",'SmallX')
    table([['Counterfactual / matched control','Validation accuracy'],['Normal labels, E=3, 120 rounds',pct(controls['clean_final']['val_accuracy'])],['Shuffled training labels, E=3, 120 rounds',pct(controls['shuffled_final']['val_accuracy'])],['Normal labels, E=3, round 40',pct(controls['clean_E3_round40']['val_accuracy'])],['Normal labels, E=10, round 40',pct(controls['large_E_final']['val_accuracy'])],['Learning rate=0 on real writers, 3 rounds','Validation loss and accuracy remain exactly fixed']],[325,167])
    p('The shuffled-label control preserves local label imbalance while removing sample-level correspondence. Its failure to approach the clean-label model supports the interpretation that normal training uses image information. I do not compare it to an assumed 1/62 chance rate: the actual class distribution is imbalanced, so a majority-class/prior strategy can exceed that number.')
    p('Increasing local epochs improves the observed 40-round result. This does not prove that client drift is absent, or that E=10 is more efficient: it uses over three times as many local passes. I treat the high-E run as a stress intervention, not an equal-compute performance claim, and retain its client-update norms/alignment records for inspection.')
    page();h('7. Non-standard diagnostics: agreement, influence and coverage')
    img(representative+'_alignment.png',height=200)
    img('heterogeneity_coverage.png',height=205)
    p('Figure 6. Top: each point is a selected client update; its y-coordinate is cosine alignment with the resulting global update, its color is local update L2 norm, and its area reflects its sample weight. Negative cosine indicates disagreement with the aggregate direction. The plot makes a large influential disagreeing update visible even when accuracy barely changes. Zero-length updates use cosine 0, a convention rather than directional evidence. Bottom: writer-label heterogeneity versus local sample size, and actual participation counts after 120 rounds.','SmallX')
    p('This is my non-standard diagnostic for serial FedAvg. I save the same plot and full numerical records for Ray and every noise scale, so I can distinguish changes in accuracy from changes in update agreement. Alignment is descriptive: agreement among clients can also mean they all follow a biased class prior, and high update norms alone do not prove harmful drift.')
    cov=r['participation'];p(f"For seed 17, C=0.04 selects each writer between {cov['C0.04']['min']} and {cov['C0.04']['max']} times, with {cov['C0.04']['unselected']} never selected. C=0.10 selects writers between {cov['C0.10']['min']} and {cov['C0.10']['max']} times, with {cov['C0.10']['unselected']} never selected. I retain the per-writer counts rather than assuming uniform participation was achieved.")
    page();h('8. Differential privacy experiment: noise and utility')
    p('For each local training image I draw independent Laplace(0,b) noise at every pixel and add it to the [0,1] image. I fix one noise draw per image/client for a run, reuse it across rounds, and leave validation and held-out images clean. This is a fixed perturbed local dataset, not fresh noise at every optimizer step. I keep the same seeds, C=0.10, E='+str(chosen['epochs'])+', minibatches and 120-round budget across all five scales.')
    img('noise_utility.png',height=245)
    p('Figure 7. Final training accuracy on perturbed inputs and held-out accuracy on clean inputs versus b; means and sample SD across three training seeds. This is the requested noise-scale plot.','SmallX')
    rows=[['b','Final noisy-train accuracy','Clean validation accuracy','Held-out accuracy']]
    for row in r['noise']:rows.append([row['scale'],summary(row['train_accuracy']),summary(row['val_accuracy']),summary(row['test_accuracy'])])
    table(rows,[35,153,152,152])
    p(('I choose b='+str(b)+' using the predeclared rule: the largest nonzero scale whose mean validation accuracy is within two percentage points of b=0. The selected model has held-out accuracy '+summary(noise['test_accuracy'])+'.') if noise else 'No nonzero scale satisfies my predeclared two-percentage-point validation utility tolerance. I therefore do not invent a qualifying scale or select one from favorable test results.')
    p('The realized trend need not be strictly monotone at every tested b, because optimization and finite samples vary. I interpret the full table and seed variability, rather than forcing a smooth curve or claiming that every increase in noise must worsen one particular run.')
    page();h('9. Noise edge cases and a non-standard perturbation diagnostic')
    selected_noise=next(row for row in s['noise_sweep'] if row['scale']==(b if b is not None else .1))['runs'][0]
    img(selected_noise+'_curves.png',height=205)
    img('noise_samples.png',height=125)
    p('Figure 8. Above: required per-round training and clean-validation loss/accuracy for the selected nonzero scale (or b=0.1 when none qualifies). Below: the same training image at all scales. Display rendering clips intensities to [0,1] for visibility; the actual training arrays do not.','SmallX')
    rows=[['b','Pixels outside [0,1]','Mean absolute added noise']]
    for row in r['noise_boundary']:rows.append([row['b'],pct(row['outside_unit_interval_fraction']),f"{row['mean_absolute_noise']:.4f}"])
    table(rows,[65,210,217])
    p('This additional diagnostic uses 64 fixed local training images and a fixed perturbation seed. It exposes a boundary effect hidden by an attractive noisy-image display: many background pixels lie near the clean domain boundary, so even small symmetric noise can leave [0,1]. I do not silently clamp those values, because that would change the assigned X\' = X + noise experiment. The b=0 identity and Laplace moment control check the endpoints and scale separately.')
    page();h('10. Parallel and noisy update agreement')
    img(ray_name+'_alignment.png',height=220)
    img(selected_noise+'_alignment.png',height=220)
    p('Figure 9. My non-standard client agreement/influence plots for the Ray run and a noisy run. Both use the definition in Figure 6, so all three assignment components have an inspectable update-level diagnostic. Colors and marker sizes are normalized within each plot; I compare numerical norms from JSONL rather than reading cross-panel color alone.','SmallX')
    p('I save each client\'s selected round, sample count, optimizer-step count, local/global cosine, local-update norm and actual start/finish timestamps. Reading the first ten rounds isolates the interaction between sampling, local optimization, weighting and aggregate movement. For a failure or collapse, I can see whether it follows a dominant writer or broadly aligned but uninformative updates instead of relying on a final accuracy scalar.')
    page();h('11. Quantitative reading of my non-standard plots')
    p('I summarize the exact JSONL points underlying Figures 6 and 9 using fixed descriptive windows: rounds 1-10 and 111-120. All rows below use representative seed 17; they are descriptive within-run statistics, not estimates of across-seed uncertainty. Weighted cosine is the mean over rounds of the client-sample-weighted cosine. Negative share counts individual selected updates with cosine below zero; median norm is the L2 size of the local parameter update.')
    diagnostic_names=[('Serial C=.10','course/'+representative),('Ray C=.04','course/'+ray_name),('Noise b='+str(b if b is not None else .1),'course/'+selected_noise)]
    rows=[['Run / rounds','Weighted cosine','Negative updates','Median L2 norm']]
    for label,key in diagnostic_names:
        for window in ['first_10','last_10']:
            q=diagnostics[key][window]
            rows.append([f"{label}: {q['first_round']}-{q['last_round']}",f"{q['sample_weighted_mean_alignment']:.3f}",pct(q['negative_update_fraction']),f"{q['median_update_norm']:.3f}"])
    table(rows,[177,100,106,109])
    early=diagnostics['course/'+representative]['first_10'];late=diagnostics['course/'+representative]['last_10']
    noisy=diagnostics['course/'+selected_noise]['last_10']
    p(f"In serial FedAvg, weighted alignment falls from {early['sample_weighted_mean_alignment']:.3f} to {late['sample_weighted_mean_alignment']:.3f}, while median local norm grows {late['median_update_norm']/early['median_update_norm']:.2f} times. The negative-update share falls from {pct(early['negative_update_fraction'])} to {pct(late['negative_update_fraction'])}. Thus weaker positive agreement does not mean more opposite-direction updates; the directions are less aligned but mostly still positive. Local updates have not become negligible, so these diagnostics do not establish convergence.")
    p(f"The noisy C=.10 run has late alignment {noisy['sample_weighted_mean_alignment']:.3f} versus clean {late['sample_weighted_mean_alignment']:.3f}, and a {noisy['median_update_norm']/late['median_update_norm']:.2f}-times larger median norm. This matched-seed, matched-client-schedule contrast is consistent with noisier local optimization; it does not prove that noise caused a specific accuracy change. Ray's higher cosine must not be called better learning: its aggregate contains four clients rather than ten, changing each client's self-contribution to the reference direction. Its correctly matched serial model has identical parameters and trajectories.")
    p('Local-epoch stress at the same 40-round horizon','Sub')
    small=diagnostics['course/fedavg_C0.1_E3_s17']['first_40'];large=diagnostics['course_controls/large_E10']['first_40']
    table([['E / rounds','Weighted cosine','Negative updates','Median L2 norm'],['3 / 1-40',f"{small['sample_weighted_mean_alignment']:.3f}",pct(small['negative_update_fraction']),f"{small['median_update_norm']:.3f}"],['10 / 1-40',f"{large['sample_weighted_mean_alignment']:.3f}",pct(large['negative_update_fraction']),f"{large['median_update_norm']:.3f}"]],[177,100,106,109])
    p(f"E=10 increases the median local norm {large['median_update_norm']/small['median_update_norm']:.2f} times and slightly reduces alignment, yet improves round-40 validation accuracy from {pct(controls['clean_E3_round40']['val_accuracy'])} to {pct(controls['large_E_final']['val_accuracy'])}. Larger local movement therefore is not, by itself, evidence of harmful drift. E=10 also uses more optimizer steps. diagnostics.py exports every run/window to nonstandard_diagnostics.json and .csv, including sample-weighted negative shares and 90th-percentile norms; it reads no test images.",'SmallX')
    page();h('12. What the privacy experiment does and does not establish')
    p('I implement the specified Laplace input-perturbation experiment. A scale b by itself is not a complete formal privacy claim: the neighboring-data definition and L1 sensitivity must also be specified. Conditional on a fixed public label, replacing any one [0,1] 784-pixel vector can change its L1 norm by at most 784, yielding the elementary image-release bound epsilon=784/b for positive b. The utility-preserving scales here therefore do not automatically imply a useful small-epsilon guarantee.')
    p('The training labels and clean evaluation summaries are not privatized. I consequently do not claim joint image-label or user-level differential privacy for this pipeline. Reusing a single perturbed dataset makes repeated model computation post-processing of that dataset under an appropriate threat model, but the complete release includes other unprotected information. Reproducible public random seeds are an experimental tool, not a deployment mechanism for secret privacy randomness.')
    p('My noise recommendation is an empirical utility choice under an explicit tolerance. A real privacy recommendation would need a stated adversary, adjacency, label treatment, release policy and defensible privacy accounting. I would not equate better-looking perturbation images with protected local data. [2]')
    p('Limits of my evidence','Sub')
    p('I use one fixed 80/20 split, three training seeds for serial/noisy sweeps, and one full Ray seed. Validation choices can overfit this split. I evaluate a small two-layer model under a finite round budget and do not claim convergence or state-of-the-art EMNIST accuracy. I report micro accuracy because the assignment requests classification accuracy, and also retain the 10th-percentile writer validation accuracy to expose heterogeneity.')
    p(f"For the selected serial combination, the final 10th-percentile client validation accuracy is {summary(best['val_client_p10'])}. This is a lower-tail writer measure, not a confidence bound. Test images are the official aggregated partition; writer-specific test performance cannot be recovered because that file does not preserve writer identities.")
    page();h('13. My additional-three-months validation plan')
    p('Month 1: reproducibility and fair optimization budgets','Sub')
    p('I would repeat at least five training seeds and several writer-preserving 80/20 splits, establish uncertainty intervals across trained models, and compare learning-rate schedules with validation-only stopping. I would compare C/E settings at equal communication bytes, equal local gradient steps and equal wall time separately. A small centralized oracle and one-client/all-client synthetic controls would test where FedAvg changes from exact gradient equivalence to heterogeneous local drift.')
    p('Month 2: non-IID failure modes and privacy threat models','Sub')
    p('I would vary writer label skew and sample imbalance without transferring samples between real clients, study rare-class and lower-tail writer behavior, and simulate unavailable or slow clients. I would compare fixed versus newly resampled image perturbations, inspect out-of-range pixels, and explicitly specify record-level versus user-level privacy and label sensitivity. With appropriate clipping and accounting, I would compare formally calibrated mechanisms and empirical reconstruction/membership attacks; an attack failure alone would not prove differential privacy.')
    p('Month 3: systems and final generalization','Sub')
    p('I would benchmark serial and Ray runs on an otherwise idle machine with fixed client schedules, record process startup/memory overhead, and vary actor lifetime/resources while checking model equality. I would test interrupted execution, deterministic checkpoint resume, and round failure recovery. After freezing the final protocol I would evaluate once on a new untouched test set, inspect per-class confusion and calibration, and publish negative results as well as successful runs.')
    p('These proposed studies target the current evidence gaps. They do not retroactively change my reported test results or imply that an unexecuted experiment passed.')
    page();h('14. How I reproduce the runs and validation')
    p('GitHub contains source, histories, client traces, figures, validation summaries and this PDF, but no private course data or checkpoint binaries. After a fresh clone, obtain the Canvas data and run experiments.py train before validate_crosschecks.py or final test evaluation. No external checkpoint download is assumed. Data provenance and checkpoint hashes are retained.')
    table([['Command from the repository root','What it does'],['python -m pip install -r requirements.txt','Install standard Python dependencies; Python 3.12'],['python prepare_data.py --train DATA/train_data.npy --test DATA/test_data.npy --pixel-divisor 1 --trusted-course-npy --output data/prepared','Load the trusted Canvas archive and reproduce the fixed writer splits'],['python validate.py --ray','Rerun synthetic identities, noise moments, learning and parallel equality'],['python validate_adapters.py; python validate_edges.py','Reproduce adapter/index checks and the boundary controls in Appendix A'],['python validate_course.py --data data/prepared','Rerun real-data label permutation, high-E and zero-rate controls'],['python experiments.py train --data data/prepared --out results/course --rounds 120','Run validation-only C/E/noise sweeps and the 4-actor Ray run; freeze selection'],['python validate_crosschecks.py','Compare all three b=0/clean checkpoint pairs and full metric histories'],['python diagnostics.py','Recompute every non-standard plot window from retained JSONL'],['python experiments.py test --data data/prepared --out results/course','Evaluate only the frozen, predeclared set of test models'],['python plots.py --data data/prepared --results results/course','Render required per-round and noise plots'],['python analysis.py --data data/prepared --results results','Compute report tables and diagnostic figures'],['python report.py --repo-url '+repo_url+' --font-dir FONT_DIR','Build this PDF from complete measured results and licensed Times New Roman fonts']],[290,202])
    p('To rerun training, use fresh output directories or archive the original results first; scripts refuse to overwrite completed experiments or final test evaluations. For a no-retraining PDF rebuild, use the retained figures and result JSON. Read README and docs/VALIDATION.md for exact assumptions, predictions and retained evidence.','SmallX')
    p('References','Sub')
    p('[1] McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. https://proceedings.mlr.press/v54/mcmahan17a.html<br/>[2] Dwork and Roth. The Algorithmic Foundations of Differential Privacy. 2014. https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf<br/>[3] Ray Core: Actors and Resources. https://docs.ray.io/en/latest/ray-core/actors.html<br/>[4] Caldas et al. LEAF: A Benchmark for Federated Settings. arXiv:1812.01097 (2018). Dataset provenance: course Canvas Assignment3-data.zip.','SmallX')
    page();h('Appendix A. Executed boundary-case evidence')
    p('These additional controls were executed on November 18, 2026 against the unchanged training and data-loader source. Every row uses a small synthetic fixture or an explicitly invalid input; none is an EMNIST accuracy result or a rerun of the full training sweep. The tests exercise the production Client, prepare, Config, fedavg and perturb functions.')
    rows=[['Boundary case','Test method','Expected behavior','Measured result']]
    rows += [[q['case'],q['method'],q['expected'],q['observed']] for q in edges['checks']]
    table(rows,[83,143,133,133])
    p('Reproduce with python validate_edges.py. Full observations, runtime versions and SHA-256 hashes of the exercised source are in results/validation/edge_validation.json; the execution log is docs/execution/edge_validation.log. A failing assertion stops the script before a passing result is written. The fixture uses one writer to isolate loader boundaries; the separate adapter validation checks the required 100-client schema.','SmallX')
    p('I also reran the three retained clean/b=0 checkpoint comparisons and the adapter validation for this revision. These checks support specific invariants; they do not independently reproduce every original training trajectory or establish a formal privacy guarantee.','SmallX')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c4cedb'));canvas.line(54,42,558,42)
        canvas.setFont('TNR',8);canvas.setFillColor(colors.HexColor('#46556a'));canvas.drawString(54,29,'Diego HW3 | Dorrie-ge | Federated Learning and Differential Privacy');canvas.drawRightString(558,29,str(doc.page))
    while story and isinstance(story[-1],Spacer):
        story.pop()
    SimpleDocTemplate(str(BASE/'report.pdf'),pagesize=letter,rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=54,title='Assignment 3 - Federated Learning and Differential Privacy',author='Dorrie-ge').build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--repo-url',required=True);a.add_argument('--font-dir',default='/System/Library/Fonts/Supplemental')
    args=a.parse_args();build(args.repo_url,args.font_dir)
