export const meta = {
    name: 'run-datamaps',
    description: 'Run datamaps pipeline inside Docker, given a PDF or PNG URL.',
    phases: [{ title: 'Download' }, { title: 'Pipeline' }],
  };

  const src = args.input;   // harness will provide this argument

  phase('Download');
  if (src.startsWith('http://') || src.startsWith('https://')) {
    const parts = src.split('/');
    const filename = parts[parts.length - 1];
    await agent(`curl -L --fail ${src} -o report/${filename}`, { label: 'download', phase: 'Download' });
  } else {
    // Assume local path relative to repo root
    await agent(`cp ${src} report/$(basename "${src}")`, { label: 'copy', phase: 'Download' });
  }

  phase('Pipeline');
  await agent(
    `python -m datamaps extract && python -m datamaps ocr && python -m datamaps split && python -m datamaps compile`,
    { label: 'run', phase: 'Pipeline' }
  );

  return {
    dataset: `data/pdf_full_dataset.json`,
    csvReport: `report/final_heatmap_report.csv`,
  };

