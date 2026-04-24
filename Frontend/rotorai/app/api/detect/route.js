export async function POST(req) {
  const body = await req.json();

  const res = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + '/detect', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.API_SECRET, // hidden
    },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  return Response.json(data);
}
