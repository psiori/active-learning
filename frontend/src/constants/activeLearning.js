export const SELECTED_PROJECT_STORAGE_KEY = 'crid-active-learning:selected-project'
export const SESSION_STORAGE_KEY = 'crid-active-learning:session:v1'

export const JOB_STAGE_LABELS = {
  query_pool: 'Query Pool',
  query_labeled: 'Query Labeled',
  fetch_seeded: 'Fetch Seeded',
  time_gap_filter: 'Time Gap Filter',
  brightness_filter: 'Brightness Filter',
  load_model: 'Load Model',
  select_samples: 'Select Samples',
  materialize_preview: 'Prepare Preview',
}

export const QUERY_STAGE_IDS = [
  'query_pool',
  'query_labeled',
  'fetch_seeded',
  'time_gap_filter',
  'brightness_filter',
]

export const STRATEGY_STAGE_IDS = [
  'query_pool',
  'query_labeled',
  'fetch_seeded',
  'time_gap_filter',
  'brightness_filter',
  'load_model',
  'select_samples',
  'materialize_preview',
]
