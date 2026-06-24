<template>
  <div class="app-root">
    <header class="app-header">
      <p class="app-kicker">CRID</p>
      <h1>Sama Import</h1>
      <p class="app-subtitle">Label import, mask generation, and dataset review</p>
    </header>

    <div class="sama-page">
      <aside class="sama-sidebar">
        <ProjectMultiSelect
          :projects="projects"
          :selected="selectedOrigins"
          :loading="loadingProjects"
          @update:selected="onSelectedOrigins"
        />
        <PipelineRunner
          :selected-batches="selectedBatches"
          @job-finished="onJobFinished"
        />
      </aside>

      <main class="sama-main">
        <div v-if="errorMessage" class="error-banner">
          {{ errorMessage }}
        </div>
        <BatchTable
          :batches="batches"
          :selected="selectedBatches"
          :loading="loadingBatches"
          :days="days"
          :enrich="enrich"
          @update:selected="selectedBatches = $event"
          @update:days="days = $event"
          @update:enrich="enrich = $event"
          @update:chosen-origin="onChosenOrigin"
          @refresh="refreshBatches"
        />
        <ReviewPanel
          :topics="reviewTopics"
          @topics-changed="reviewTopics = $event"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue'
import ProjectMultiSelect from './components/ProjectMultiSelect.vue'
import BatchTable from './components/BatchTable.vue'
import PipelineRunner from './components/PipelineRunner.vue'
import ReviewPanel from './components/ReviewPanel.vue'
import { fetchSamaProjects, fetchSamaBatches } from './api/samaApi'

const projects = ref([])
const loadingProjects = ref(false)
const selectedOrigins = ref([])

const batches = ref([])
const selectedBatches = ref([])
const loadingBatches = ref(false)
const days = ref(90)
const enrich = ref(true)

const reviewTopics = ref([])
const errorMessage = ref('')

onMounted(async () => {
  loadingProjects.value = true
  try {
    projects.value = await fetchSamaProjects()
  } finally {
    loadingProjects.value = false
  }
  selectedOrigins.value = [...new Set(projects.value.map((p) => p.origin))]
  await refreshBatches()
})

async function refreshBatches() {
  loadingBatches.value = true
  errorMessage.value = ''
  try {
    const result = await fetchSamaBatches({
      origins: selectedOrigins.value,
      days: days.value,
      enrich: enrich.value,
    })
    const incoming = result.batches || []
    for (const b of incoming) {
      b.chosen_origin = b.origin
    }
    batches.value = incoming
    const keys = new Set(incoming.map((b) => `${b.project_id}-${b.batch_id}`))
    selectedBatches.value = selectedBatches.value.filter((b) =>
      keys.has(`${b.project_id}-${b.batch_id}`),
    )
  } catch (e) {
    console.error(e)
    errorMessage.value = e?.message || 'Failed to load batches.'
    batches.value = []
  } finally {
    loadingBatches.value = false
  }
}

function onChosenOrigin({ batch, origin }) {
  const target = batches.value.find(
    (b) => b.project_id === batch.project_id && b.batch_id === batch.batch_id,
  )
  if (target) target.chosen_origin = origin
  const sel = selectedBatches.value.find(
    (b) => b.project_id === batch.project_id && b.batch_id === batch.batch_id,
  )
  if (sel) sel.chosen_origin = origin
}

function onSelectedOrigins(next) {
  selectedOrigins.value = next
}

watch([selectedOrigins, days, enrich], () => {
  // No auto-refresh to avoid surprise Sama API hits; user clicks refresh.
})

function onJobFinished(job) {
  const result = job?.result
  if (!result) return
  const topics = new Set(reviewTopics.value)
  const collect = (mask) => {
    if (!mask) return
    if (Array.isArray(mask.dataset_topics)) {
      for (const t of mask.dataset_topics) topics.add(t)
    } else if (mask.dataset_name) {
      topics.add(mask.dataset_name)
    }
  }
  if (Array.isArray(result.runs)) {
    for (const run of result.runs) {
      collect(run.result?.masks)
      const topicsFromImport = run.result?.import?.dataset_topics
      if (Array.isArray(topicsFromImport)) {
        for (const t of topicsFromImport) {
          if (typeof t === 'string' && !t.startsWith('[dry-run]')) topics.add(t)
        }
      }
    }
  }
  reviewTopics.value = [...topics]
  refreshBatches()
}
</script>

<style scoped>
.app-root {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  padding: 1rem 1.25rem 0.5rem;
  background: #1f2937;
  color: #fff;
}
.app-kicker {
  margin: 0;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
}
.app-header h1 {
  margin: 0.15rem 0 0;
  font-size: 1.25rem;
}
.app-subtitle {
  margin: 0.25rem 0 0;
  font-size: 0.85rem;
  color: #cbd5e1;
}
.sama-page {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 1rem;
  padding: 1rem;
  flex: 1;
}
.sama-sidebar {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.sama-main {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}
.error-banner {
  background: #fde2e2;
  color: #7a1f1f;
  padding: 0.6rem 0.9rem;
  border-radius: 8px;
  font-size: 0.85rem;
  white-space: pre-wrap;
}
@media (max-width: 1024px) {
  .sama-page {
    grid-template-columns: 1fr;
  }
}
</style>
