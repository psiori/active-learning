<template>
  <section class="sama-card review-card">
    <header>
      <h2>Review</h2>
      <p class="hint">
        Inspect mosaics and approve dataset topics that look clean. Approval writes
        an APPROVED marker file in the topic cache dir.
      </p>
    </header>

    <div class="topic-controls">
      <input
        v-model="manualTopic"
        type="text"
        placeholder="Add a dataset topic to review…"
        @keyup.enter="addTopic"
      />
      <button class="primary" @click="addTopic" :disabled="!manualTopic">Add</button>
    </div>

    <div v-if="!topics.length" class="empty">
      No topics queued. Run the pipeline or paste a topic name above.
    </div>

    <article v-for="topic in topics" :key="topic" class="topic">
      <header class="topic-header">
        <h3>{{ topic }}</h3>
        <div class="topic-actions">
          <span
            v-if="states[topic]?.approved"
            class="badge approved"
          >approved</span>
          <span
            v-else-if="states[topic]?.masks_exist"
            class="badge masks-ready"
          >masks ready</span>
          <span v-else class="badge">unknown</span>
          <button
            v-if="!states[topic]?.approved"
            class="primary"
            :disabled="!states[topic]?.masks_exist"
            @click="setApproved(topic, true)"
          >Approve</button>
          <button
            v-else
            class="secondary"
            @click="setApproved(topic, false)"
          >Revoke</button>
          <button class="link" @click="reload(topic)">↻</button>
        </div>
      </header>
      <div class="mosaic-wrap">
        <img
          v-if="states[topic]?.masks_exist"
          :src="mosaicUrl(topic) + `?ts=${reloadKey[topic] ?? 0}`"
          :alt="`Mosaic for ${topic}`"
        />
        <p v-else class="empty">No mosaic available yet. Generate masks first.</p>
      </div>
    </article>
  </section>
</template>

<script setup>
import { reactive, ref, watch } from 'vue'
import { approveTopic, fetchTopicState, mosaicUrl } from '../api/samaApi'

const props = defineProps({
  topics: { type: Array, default: () => [] },
})
const emit = defineEmits(['topics-changed'])

const manualTopic = ref('')
const states = reactive({})
const reloadKey = reactive({})

async function loadState(topic) {
  try {
    states[topic] = await fetchTopicState(topic)
  } catch (e) {
    states[topic] = { error: e.message }
  }
}

watch(
  () => props.topics,
  (next) => {
    for (const t of next) {
      if (!(t in states)) loadState(t)
    }
  },
  { immediate: true, deep: true },
)

function addTopic() {
  if (!manualTopic.value) return
  const t = manualTopic.value.trim()
  if (!props.topics.includes(t)) {
    emit('topics-changed', [...props.topics, t])
  }
  manualTopic.value = ''
}

function reload(topic) {
  reloadKey[topic] = (reloadKey[topic] ?? 0) + 1
  loadState(topic)
}

async function setApproved(topic, approved) {
  try {
    await approveTopic(topic, approved)
    await loadState(topic)
  } catch (e) {
    states[topic] = { ...(states[topic] || {}), error: e.message }
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
.review-card header h2 {
  margin: 0;
  font-size: 1rem;
}
.hint {
  font-size: 0.8rem;
  color: #666;
  margin: 0.25rem 0 0.75rem;
}
.topic-controls {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.topic-controls input {
  flex: 1;
  padding: 0.4rem 0.5rem;
  border: 1px solid #ccc;
  border-radius: 6px;
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
.link {
  background: none;
  border: none;
  color: #3366cc;
  cursor: pointer;
  font-size: 1rem;
}
.empty {
  color: #888;
  font-size: 0.9rem;
  text-align: center;
  padding: 1rem;
}
.topic {
  border-top: 1px solid #eee;
  padding-top: 0.75rem;
  margin-top: 0.75rem;
}
.topic-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
}
.topic-header h3 {
  margin: 0;
  font-size: 0.95rem;
  font-family: monospace;
}
.topic-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.badge {
  font-size: 0.75rem;
  padding: 0.1rem 0.5rem;
  border-radius: 999px;
  background: #eee;
  color: #444;
}
.badge.approved {
  background: #c8e6c9;
  color: #1b5e20;
}
.badge.masks-ready {
  background: #fff3cd;
  color: #7c5d00;
}
.mosaic-wrap img {
  width: 100%;
  border-radius: 6px;
  border: 1px solid #ddd;
}
</style>
