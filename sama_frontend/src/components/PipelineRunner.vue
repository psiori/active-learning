<template>
  <section class="sama-card runner-card">
    <header>
      <h2>Run pipeline</h2>
      <p class="hint">
        Sequential per project: import → masks. Review afterwards in the panel below.
      </p>
    </header>

    <div class="options">
      <label>
        <input type="checkbox" v-model="dryRun" />
        Dry run
      </label>
      <label>
        <input type="checkbox" v-model="importInProgress" />
        Include in-progress tasks
      </label>
      <label>
        <input type="checkbox" v-model="redoMasks" />
        Re-generate masks (overwrite)
      </label>
    </div>

    <div class="grouped">
      <p v-if="!groups.length" class="empty">
        Select one or more batches above to enable pipeline runs.
      </p>
      <ul v-else>
        <li v-for="g in groups" :key="`${g.origin}-${g.project_id}`">
          <strong>{{ g.origin }} / {{ g.project_type }}</strong>
          — {{ g.batch_ids.length }} batch(es): {{ g.batch_ids.join(', ') }}
        </li>
      </ul>
    </div>

    <div class="actions">
      <button
        class="primary"
        :disabled="!groups.length || running"
        @click="runPipeline"
      >
        {{ running ? 'Running…' : 'Run pipeline' }}
      </button>
      <button class="secondary" :disabled="!running" @click="$emit('cancel')">
        Cancel job
      </button>
    </div>

    <div v-if="job" class="job-view">
      <p>
        <strong>Job:</strong> {{ job.job_id }} —
        <em>{{ job.state }}</em>
        <span v-if="job.stage"> · stage: {{ job.stage }}</span>
      </p>
      <p v-if="job.message" class="message">{{ job.message }}</p>
      <p v-if="job.error" class="error">{{ job.error.type }}: {{ job.error.detail }}</p>
      <details v-if="job.result" open>
        <summary>Result</summary>
        <pre>{{ JSON.stringify(job.result, null, 2) }}</pre>
      </details>
    </div>
  </section>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { startPipelineJob, openJobEventStream } from '../api/samaApi'

const props = defineProps({
  selectedBatches: { type: Array, default: () => [] },
})
const emit = defineEmits(['job-finished', 'cancel'])

const dryRun = ref(false)
const importInProgress = ref(false)
const redoMasks = ref(false)
const running = ref(false)
const job = ref(null)
let source = null

const groups = computed(() => {
  const map = new Map()
  for (const batch of props.selectedBatches) {
    const origin = batch.chosen_origin || batch.origin
    const key = `${origin}-${batch.project_id}`
    if (!map.has(key)) {
      map.set(key, {
        origin,
        project_id: batch.project_id,
        project_type: batch.project_type,
        batch_ids: [],
      })
    }
    map.get(key).batch_ids.push(batch.batch_id)
  }
  return [...map.values()]
})

watch(
  () => props.selectedBatches,
  () => {
    if (!running.value) job.value = null
  },
)

async function runPipeline() {
  if (!groups.value.length) return
  closeStream()
  running.value = true
  job.value = null
  try {
    const payload = {
      redo_masks: redoMasks.value,
      runs: groups.value.map((g) => ({
        origin: g.origin,
        project_id: g.project_id,
        batch_ids: g.batch_ids,
        dry_run: dryRun.value,
        import_in_progress: importInProgress.value,
      })),
    }
    const started = await startPipelineJob(payload)
    streamJob(started.job_id)
  } catch (e) {
    job.value = { error: { type: 'startup', detail: e.message } }
    running.value = false
  }
}

function streamJob(jobId) {
  source = openJobEventStream(jobId, {
    onEvent(_name, data) {
      job.value = { ...data }
    },
    onEnd() {
      running.value = false
      emit('job-finished', job.value)
    },
    onError() {
      running.value = false
    },
  })
}

function closeStream() {
  if (source) {
    source.close()
    source = null
  }
}
</script>

<style scoped>
.sama-card {
  background: var(--surface-card, #fff);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.runner-card header h2 {
  margin: 0;
  font-size: 1rem;
}
.hint {
  font-size: 0.8rem;
  color: #666;
  margin: 0.25rem 0 0.75rem;
}
.options {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 0.5rem 0 1rem;
  font-size: 0.85rem;
}
.grouped ul {
  margin: 0.25rem 0;
  padding-left: 1.25rem;
  font-size: 0.85rem;
}
.empty {
  color: #888;
  font-size: 0.85rem;
}
.actions {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.75rem;
}
.primary {
  background: #3366cc;
  color: #fff;
  border: none;
  padding: 0.4rem 0.9rem;
  border-radius: 6px;
  cursor: pointer;
}
.primary:disabled {
  background: #aaa;
  cursor: not-allowed;
}
.secondary {
  background: #eee;
  color: #333;
  border: none;
  padding: 0.4rem 0.9rem;
  border-radius: 6px;
  cursor: pointer;
}
.job-view {
  margin-top: 1rem;
  background: #f7f7f9;
  padding: 0.75rem;
  border-radius: 8px;
  font-size: 0.85rem;
}
.message {
  color: #333;
}
.error {
  color: #b00020;
}
pre {
  background: #fff;
  padding: 0.5rem;
  overflow: auto;
  max-height: 250px;
  font-size: 0.75rem;
}
</style>
