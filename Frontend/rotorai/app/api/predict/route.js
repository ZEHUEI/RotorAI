export async function POST(req) {
  const formData = await req.formData();

  const res = await fetch(process.env.BACKEND_URL + '/predict', {
    method: 'POST',
    body: formData,
    headers: {
      'x-api-key': process.env.API_SECRET, // hidden
    },
  });

  const data = await res.json();
  return Response.json(data);
}
