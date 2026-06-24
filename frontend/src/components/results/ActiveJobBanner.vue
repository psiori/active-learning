<template>
  <section class="active-job-banner">
    <div class="active-job-banner__row">
      <p class="active-job-banner__label">{{ stageLabel }}</p>
      <p class="active-job-banner__meta">
        <span v-if="activeJob.completed != null && activeJob.total != null">
          {{ activeJob.completed }} / {{ activeJob.total }}
        </span>
        <span v-else>
          {{ activeJob.state }}
        </span>
      </p>
    </div>
    <p class="active-job-banner__message">{{ activeJob.message || 'Working…' }}</p>
    <div class="active-job-banner__track">
      <div class="active-job-banner__fill" :style="{ width: `${activeJobPercent}%` }" />
    </div>
    <div class="active-job-banner__stages">
      <div v-for="stage in activeJobStages" :key="stage.id" class="active-job-banner__stage-row">
        <span>{{ stage.label }}</span>
        <strong :class="`stage-badge stage-badge--${stage.state}`">{{ stage.state }}</strong>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { JOB_STAGE_LABELS } from '@/constants/activeLearning'

const props = defineProps({
  activeJob: {
    type: Object,
    required: true,
  },
  activeJobPercent: {
    type: Number,
    required: true,
  },
  activeJobStages: {
    type: Array,
    required: true,
  },
})

const stageLabel = computed(() => JOB_STAGE_LABELS[props.activeJob.stage] || 'Queued')
</script>
