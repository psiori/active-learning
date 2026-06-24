export function parseTimestampFromSampleId(sampleId) {
  if (!sampleId) return null
  const source = sampleId.split('/').pop() || sampleId
  const isoMatch = source.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})(?:_(\d{3}))?Z/i)
  if (isoMatch) {
    const [, year, month, day, hour, minute, second, millis] = isoMatch
    return {
      date: `${year}-${month}-${day}`,
      time: `${hour}:${minute}:${second} UTC`,
      epochMs: Date.UTC(
        Number(year),
        Number(month) - 1,
        Number(day),
        Number(hour),
        Number(minute),
        Number(second),
        millis ? Number(millis) : 0,
      ),
    }
  }

  const compactMatch = source.match(/(\d{6})_(\d{6,7})/)
  if (!compactMatch) {
    return null
  }

  const [, rawDate, rawTime] = compactMatch
  const year = `20${rawDate.slice(0, 2)}`
  const month = rawDate.slice(2, 4)
  const day = rawDate.slice(4, 6)
  const hour = rawTime.slice(0, 2)
  const minute = rawTime.slice(2, 4)
  const second = rawTime.slice(4, 6)
  const millis = rawTime.length > 6 ? rawTime.slice(6) : '0'

  return {
    date: `${year}-${month}-${day}`,
    time: `${hour}:${minute}:${second}`,
    epochMs: Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
      Number(millis),
    ),
  }
}

export function sampleIdSortKey(sampleId) {
  const parsed = parseTimestampFromSampleId(sampleId)
  return parsed ? parsed.epochMs : Number.POSITIVE_INFINITY
}
