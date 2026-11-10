import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the upload form and heading', () => {
  render(<App />);
  expect(screen.getByText(/Crack Detection using DeepCrack/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /predict/i })).toBeInTheDocument();
});
